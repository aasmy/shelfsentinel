
import yaml


def load_zones(filepath="zones.yaml"):
    """Load zone polygons from a YAML file."""
    with open(filepath, "r") as file:
        zones = yaml.safe_load(file)

    return zones or {}


def save_zones(zones, filepath="zones.yaml"):
    """Save zone polygons to a YAML file."""
    with open(filepath, "w") as file:
        yaml.safe_dump(zones, file, sort_keys=False)


def point_in_polygon(point, polygon):
    """Return True if point (x, y) is inside the polygon."""
    x, y = point
    inside = False

    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if (yi > y) != (yj > y):
            intersection_x = (
                (xj - xi) * (y - yi) / (yj - yi) + xi
            )

            if x < intersection_x:
                inside = not inside

        j = i

    return inside


def get_zone(point, zones):
    """Return the zone containing a point, or None."""
    for zone_name, polygon in zones.items():
        if point_in_polygon(point, polygon):
            return zone_name

    return None
