"""A small, readable 2D rigid-body solver.

The implementation is deliberately dependency-free.  It is a teaching
reference for the Newton solver series, not a production collision library.
The solver uses:

* semi-implicit Euler integration;
* SAT collision detection for convex polygons;
* positional penetration correction;
* sequential impulses for normal and friction constraints.

Run the built-in scenarios with:

    python scripts/newton_solver_from_zero/rigid2d.py --scenario ground
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


EPSILON = 1.0e-9


@dataclass(frozen=True)
class Vec2:
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Vec2":
        if abs(scalar) < EPSILON:
            raise ZeroDivisionError("cannot divide a vector by zero")
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        """The scalar z component of the 2D cross product."""
        return self.x * other.y - self.y * other.x

    def length_squared(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def normalized(self) -> "Vec2":
        length = self.length()
        return self if length < EPSILON else self / length

    def perpendicular(self) -> "Vec2":
        return Vec2(-self.y, self.x)


def cross_scalar_vector(scalar: float, vector: Vec2) -> Vec2:
    """Return the 2D equivalent of scalar cross vector."""
    return Vec2(-scalar * vector.y, scalar * vector.x)


def rotate(vector: Vec2, angle: float) -> Vec2:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return Vec2(cosine * vector.x - sine * vector.y,
                sine * vector.x + cosine * vector.y)


@dataclass
class RigidBody:
    """A convex polygon with a single 2D rotational degree of freedom."""

    position: Vec2
    local_vertices: tuple[Vec2, ...]
    mass: float = 1.0
    inertia: float = 1.0
    angle: float = 0.0
    velocity: Vec2 = field(default_factory=Vec2)
    angular_velocity: float = 0.0
    friction: float = 0.5
    restitution: float = 0.0
    dynamic: bool = True
    force: Vec2 = field(default_factory=Vec2)
    torque: float = 0.0
    name: str = "body"

    def __post_init__(self) -> None:
        if len(self.local_vertices) < 3:
            raise ValueError("a rigid body needs at least three vertices")
        if self.dynamic and (self.mass <= 0.0 or self.inertia <= 0.0):
            raise ValueError("dynamic bodies need positive mass and inertia")
        if not self.dynamic:
            self.mass = math.inf
            self.inertia = math.inf

    @property
    def inverse_mass(self) -> float:
        return 0.0 if not self.dynamic else 1.0 / self.mass

    @property
    def inverse_inertia(self) -> float:
        return 0.0 if not self.dynamic else 1.0 / self.inertia

    @classmethod
    def box(
        cls,
        position: Vec2,
        half_extents: Vec2,
        *,
        mass: float = 1.0,
        angle: float = 0.0,
        velocity: Vec2 | None = None,
        friction: float = 0.5,
        restitution: float = 0.0,
        dynamic: bool = True,
        name: str = "box",
    ) -> "RigidBody":
        """Create a rectangle and compute its planar box inertia."""
        hx, hy = half_extents.x, half_extents.y
        vertices = (Vec2(-hx, -hy), Vec2(hx, -hy),
                    Vec2(hx, hy), Vec2(-hx, hy))
        width, height = 2.0 * hx, 2.0 * hy
        inertia = mass * (width * width + height * height) / 12.0
        return cls(
            position=position,
            local_vertices=vertices,
            mass=mass,
            inertia=inertia,
            angle=angle,
            velocity=velocity or Vec2(),
            friction=friction,
            restitution=restitution,
            dynamic=dynamic,
            name=name,
        )

    def world_vertices(self) -> tuple[Vec2, ...]:
        return tuple(self.position + rotate(vertex, self.angle)
                     for vertex in self.local_vertices)

    def velocity_at(self, world_point: Vec2) -> Vec2:
        radius = world_point - self.position
        return self.velocity + cross_scalar_vector(self.angular_velocity, radius)

    def clear_forces(self) -> None:
        self.force = Vec2()
        self.torque = 0.0

    def apply_force(self, force: Vec2, world_point: Vec2 | None = None) -> None:
        if not self.dynamic:
            return
        self.force = self.force + force
        if world_point is not None:
            self.torque += (world_point - self.position).cross(force)

    def integrate_forces(self, dt: float, gravity: Vec2) -> None:
        if not self.dynamic:
            return
        acceleration = gravity + self.force * self.inverse_mass
        self.velocity = self.velocity + acceleration * dt
        self.angular_velocity += self.torque * self.inverse_inertia * dt

    def integrate_velocity(self, dt: float) -> None:
        if not self.dynamic:
            return
        self.position = self.position + self.velocity * dt
        self.angle += self.angular_velocity * dt

    def apply_impulse(self, impulse: Vec2, world_point: Vec2) -> None:
        if not self.dynamic:
            return
        radius = world_point - self.position
        self.velocity = self.velocity + impulse * self.inverse_mass
        self.angular_velocity += radius.cross(impulse) * self.inverse_inertia


@dataclass
class Contact:
    """A contact whose normal points from body A toward body B."""

    body_a: RigidBody
    body_b: RigidBody
    point: Vec2
    normal: Vec2
    penetration: float
    normal_impulse: float = 0.0
    tangent_impulse: float = 0.0
    target_normal_velocity: float = 0.0

    @property
    def tangent(self) -> Vec2:
        return self.normal.perpendicular()

    @property
    def friction(self) -> float:
        return math.sqrt(self.body_a.friction * self.body_b.friction)

    def relative_velocity(self) -> Vec2:
        return self.body_b.velocity_at(self.point) - self.body_a.velocity_at(self.point)

    def effective_inverse_mass(self, direction: Vec2) -> float:
        radius_a = self.point - self.body_a.position
        radius_b = self.point - self.body_b.position
        lever_a = radius_a.cross(direction)
        lever_b = radius_b.cross(direction)
        return (
            self.body_a.inverse_mass
            + self.body_b.inverse_mass
            + lever_a * lever_a * self.body_a.inverse_inertia
            + lever_b * lever_b * self.body_b.inverse_inertia
        )

    def apply_impulse(self, impulse: Vec2) -> None:
        # The normal points A -> B, so A receives -impulse and B receives it.
        self.body_a.apply_impulse(-impulse, self.point)
        self.body_b.apply_impulse(impulse, self.point)


def project(vertices: Iterable[Vec2], axis: Vec2) -> tuple[float, float]:
    values = [vertex.dot(axis) for vertex in vertices]
    return min(values), max(values)


def polygon_axes(vertices: tuple[Vec2, ...]) -> Iterable[Vec2]:
    for index, vertex in enumerate(vertices):
        edge = vertices[(index + 1) % len(vertices)] - vertex
        if edge.length_squared() > EPSILON:
            yield edge.perpendicular().normalized()


def support_average(vertices: tuple[Vec2, ...], direction: Vec2) -> Vec2:
    """Average tied support vertices to avoid artificial long lever arms."""
    best = max(vertex.dot(direction) for vertex in vertices)
    candidates = [vertex for vertex in vertices
                  if best - vertex.dot(direction) < 1.0e-7]
    return Vec2(
        sum(vertex.x for vertex in candidates) / len(candidates),
        sum(vertex.y for vertex in candidates) / len(candidates),
    )


def collide(body_a: RigidBody, body_b: RigidBody) -> Contact | None:
    """SAT narrow phase for two convex polygons."""
    vertices_a = body_a.world_vertices()
    vertices_b = body_b.world_vertices()
    best_overlap = math.inf
    best_axis: Vec2 | None = None

    for axis in (*polygon_axes(vertices_a), *polygon_axes(vertices_b)):
        min_a, max_a = project(vertices_a, axis)
        min_b, max_b = project(vertices_b, axis)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap <= 0.0:
            return None
        if overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis

    if best_axis is None:
        return None

    center_delta = body_b.position - body_a.position
    if center_delta.dot(best_axis) < 0.0:
        best_axis = -best_axis

    point_a = support_average(vertices_a, best_axis)
    point_b = support_average(vertices_b, -best_axis)
    point = (point_a + point_b) * 0.5
    return Contact(body_a, body_b, point, best_axis, best_overlap)


def detect_contacts(bodies: list[RigidBody]) -> list[Contact]:
    contacts: list[Contact] = []
    for index, body_a in enumerate(bodies):
        for body_b in bodies[index + 1:]:
            if not body_a.dynamic and not body_b.dynamic:
                continue
            contact = collide(body_a, body_b)
            if contact is not None:
                contacts.append(contact)
    return contacts


def correct_positions(
    contacts: list[Contact],
    *,
    percent: float = 0.8,
    slop: float = 0.005,
) -> None:
    """Remove some overlap without pretending this is a velocity impulse."""
    for contact in contacts:
        inverse_mass_sum = (contact.body_a.inverse_mass
                            + contact.body_b.inverse_mass)
        if inverse_mass_sum < EPSILON:
            continue
        depth = max(contact.penetration - slop, 0.0)
        correction = contact.normal * (percent * depth / inverse_mass_sum)
        if contact.body_a.dynamic:
            contact.body_a.position = contact.body_a.position - correction * contact.body_a.inverse_mass
        if contact.body_b.dynamic:
            contact.body_b.position = contact.body_b.position + correction * contact.body_b.inverse_mass


def solve_velocity_constraints(
    contacts: list[Contact],
    *,
    iterations: int = 8,
    restitution_threshold: float = 1.0,
    warm_start: bool = False,
) -> None:
    """Solve normal and friction constraints with sequential impulses."""
    if not warm_start:
        for contact in contacts:
            contact.normal_impulse = 0.0
            contact.tangent_impulse = 0.0

    for contact in contacts:
        normal_speed = contact.relative_velocity().dot(contact.normal)
        if normal_speed < -restitution_threshold:
            contact.target_normal_velocity = -min(contact.body_a.restitution,
                                                   contact.body_b.restitution) * normal_speed
        else:
            contact.target_normal_velocity = 0.0

    for _ in range(iterations):
        for contact in contacts:
            normal = contact.normal
            relative = contact.relative_velocity()
            normal_speed = relative.dot(normal)
            effective_normal_inverse_mass = contact.effective_inverse_mass(normal)
            if effective_normal_inverse_mass > EPSILON:
                # A negative bias asks the impulse solver to separate overlap.
                bias = -0.2 * max(contact.penetration - 0.005, 0.0)
                delta = -(normal_speed - contact.target_normal_velocity + bias) / effective_normal_inverse_mass
                new_total = max(0.0, contact.normal_impulse + delta)
                applied = new_total - contact.normal_impulse
                contact.normal_impulse = new_total
                contact.apply_impulse(normal * applied)

            tangent = contact.tangent
            relative = contact.relative_velocity()
            tangent_speed = relative.dot(tangent)
            effective_tangent_inverse_mass = contact.effective_inverse_mass(tangent)
            if effective_tangent_inverse_mass <= EPSILON:
                continue
            delta = -tangent_speed / effective_tangent_inverse_mass
            limit = contact.friction * contact.normal_impulse
            new_total = max(-limit, min(limit, contact.tangent_impulse + delta))
            applied = new_total - contact.tangent_impulse
            contact.tangent_impulse = new_total
            contact.apply_impulse(tangent * applied)


@dataclass
class World:
    bodies: list[RigidBody]
    gravity: Vec2 = Vec2(0.0, -9.81)
    velocity_iterations: int = 8
    position_iterations: int = 2
    position_percent: float = 0.8
    position_slop: float = 0.005

    def step(self, dt: float) -> list[Contact]:
        """Advance one fixed time step and return the contacts solved."""
        for body in self.bodies:
            body.integrate_forces(dt, self.gravity)
        for body in self.bodies:
            body.integrate_velocity(dt)

        contacts = detect_contacts(self.bodies)
        for _ in range(self.position_iterations):
            correct_positions(contacts, percent=self.position_percent,
                              slop=self.position_slop)
        solve_velocity_constraints(contacts, iterations=self.velocity_iterations)

        for body in self.bodies:
            body.clear_forces()
        return contacts


def make_ground() -> RigidBody:
    return RigidBody.box(
        position=Vec2(0.0, -0.5),
        half_extents=Vec2(20.0, 0.5),
        dynamic=False,
        friction=0.8,
        restitution=0.0,
        name="ground",
    )


def run_scenario(scenario: str, steps: int, dt: float) -> list[dict[str, float | str]]:
    if scenario == "free_fall":
        bodies = [RigidBody.box(Vec2(0.0, 5.0), Vec2(0.5, 0.5), name="box")]
    elif scenario == "ground":
        bodies = [make_ground(), RigidBody.box(
            Vec2(0.0, 3.0), Vec2(0.5, 0.5), restitution=0.2, name="box")]
    elif scenario == "two_body":
        bodies = [
            RigidBody.box(Vec2(-2.0, 1.0), Vec2(0.5, 0.5),
                          velocity=Vec2(4.0, 0.0), name="left"),
            RigidBody.box(Vec2(2.0, 1.0), Vec2(0.5, 0.5),
                          velocity=Vec2(-1.0, 0.0), name="right"),
            make_ground(),
        ]
    elif scenario == "stack":
        bodies = [make_ground()]
        for index in range(4):
            bodies.append(RigidBody.box(
                Vec2(0.0, 0.55 + index * 1.01), Vec2(0.5, 0.5),
                friction=0.7, name=f"box_{index}"))
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    world = World(bodies)
    records: list[dict[str, float | str]] = []
    for step in range(steps + 1):
        time = step * dt
        for body in world.bodies:
            records.append({
                "step": float(step),
                "time": time,
                "body": body.name,
                "x": body.position.x,
                "y": body.position.y,
                "vx": body.velocity.x,
                "vy": body.velocity.y,
                "angle": body.angle,
                "omega": body.angular_velocity,
            })
        if step < steps:
            world.step(dt)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("free_fall", "ground", "two_body", "stack"), default="ground")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    records = run_scenario(args.scenario, args.steps, args.dt)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    else:
        dynamic_records = [record for record in records if record["body"] != "ground"]
        for record in dynamic_records[-min(5, len(dynamic_records)):]:
            print(
                f"t={record['time']:.3f} body={record['body']} "
                f"pos=({record['x']:.3f}, {record['y']:.3f}) "
                f"vel=({record['vx']:.3f}, {record['vy']:.3f}) "
                f"angle={record['angle']:.3f}"
            )


if __name__ == "__main__":
    main()
