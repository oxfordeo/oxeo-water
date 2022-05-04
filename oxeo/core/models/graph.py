import networkx as nx
import xarray as xa

NODE_SCHEMA = {
    "time": None,
    "id": None,
    "band": None,
    "data_coverage": None,
    "eo:cloud_cover": None,
    "sentinel:latitude_band": None,
    "sentinel:grid_square": None,
    "updated": None,
    "sentinel:product_id": None,
    "platform": None,
    "gsd": None,
    "created": None,
    "sentinel:data_coverage": None,
    "sentinel:valid_cloud_cover": None,
    "instruments": None,
    "view:off_nadir": None,
    "sentinel:utm_zone": None,
    "sentinel:sequence": None,
    "constellation": None,
    "proj:epsg": None,
    "title": None,
    "epsg": None,
    "bbox": None,
    "res_level": None,
}


def stackstac_xa_coords_to_dict(
    coords: xa.core.coordinates.DataArrayCoordinates,
) -> dict:
    """Given the coordinates of a xarray.DataArray obtained from stackstac,
    return a dictionary of the coordinates.

    Args:
        coords (xa.DataArrayCoordinates): The coordinates of the xarray.DataArray

    Returns:
        dict:  A dictionary of the coordinates
    """
    res = {}
    d = coords.to_dataset().to_dict()["coords"]
    for k, v in d.items():
        if k not in ["x", "y"]:
            data = v["data"]
            res[k] = data
    x_values = d["x"]["data"]
    y_values = d["y"]["data"]

    # Instead of saving each x and y, we just save the bbox
    res["bbox"] = list((x_values[0], y_values[-1], x_values[-1], y_values[0]))
    return res


def append_attrs(
    G: nx.Graph, node: str, attrs: dict, appendable_attrs: list = None
) -> None:
    """Append the attrs to the node in G.
    Before appending it checks if the node already exists in G and if it does,
    it appends the attrs to the existing node only if it is a list.
    Otherwise, it skips the attr.

    This function modifies the graph.

    Args:
        G (nx.Graph): The graph
        node (str): The node
        attrs (dict):  The attrs to append
        appendable_attrs (list): The attrs that can be appended

    """
    for k, v in attrs.items():
        n = G.nodes.get(node)
        if n is None:
            raise KeyError(f"Node {node} not found in graph")
        attr = G.nodes[node].get(k)
        if attr is None:
            if k in appendable_attrs:
                G.nodes[node][k] = [v]
            else:
                G.nodes[node][k] = v
        elif k in appendable_attrs:
            G.nodes[node][k].append(v)
