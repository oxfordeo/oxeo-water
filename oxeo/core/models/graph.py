import datetime
from typing import List

import networkx as nx
import xarray as xa
from gremlin_python.process.graph_traversal import GraphTraversalSource, __
from gremlin_python.process.traversal import Cardinality, Column, T

g_id = T.id
single = Cardinality.single

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


def add_edge_to_graph(
    g: GraphTraversalSource, label: str, from_id: str, to_id: str, e_properties: dict
):
    new_edge = g.V(from_id).add_e(label).to(g.V(to_id).next())
    for k, v in e_properties.items():
        if isinstance(v, list):
            for e in v:
                new_edge = new_edge.property(k, e)
        else:
            new_edge = new_edge.property(single, k, v)
    new_edge.next()


def add_revisit(
    g: GraphTraversalSource,
    constellation_node_id: str,
    aoi_node_id: str,
    v_properties: dict,
    timestamp: datetime.datetime,
):

    # Create a Transaction.
    tx = g.tx()

    # Spawn a new GraphTraversalSource, binding all traversals established from it to tx.
    gtx = tx.begin()

    try:
        # Execute a traversal within the transaction.
        add_node_to_graph(gtx, "revisit", v_properties)
        add_edge_to_graph(
            gtx,
            "captures",
            constellation_node_id,
            v_properties["id"],
            {"timestamp": timestamp},
        )
        add_edge_to_graph(gtx, "on", v_properties["id"], aoi_node_id, {})
        tx.commit()
    except Exception as e:
        # Rollback the transaction if an error occurs.
        print("Transaction failed: ", e)
        tx.rollback()


# def add_revisit(
#     g: GraphTraversalSource, parent_node: str, v_properties: dict, timestamp: str
# ):
#     add_node_to_graph(g, "revisit", v_properties)
#     g.V(parent_node).addE("has").to(g.V(v_properties["id"]).next()).property(
#         "timestamp", timestamp
#     ).next()


def add_node_to_graph(g: GraphTraversalSource, vlabel: str, properties: dict):
    v_id = properties["id"]
    g.addV(vlabel).property(T.id, v_id).next()

    for k, v in properties.items():
        if isinstance(v, list):
            for e in v:
                g.V(v_id).property(k, e).next()
        else:
            g.V(v_id).property(single, k, v).next()


def add_nodes_to_graph(g: GraphTraversalSource, entity: str, nodes: List[dict]):
    """Add given nodes to graph

    Args:
        g (GraphTraversalSource): the graph
        entity: (str): The entity of the nodes to add
        nodes (dict): A list of dict containing node properties

    Returns:
        GraphTraversalSource:  The graph
    """
    single_cardinality = {k: v for k, v in nodes.items() if not isinstance(v, list)}
    g.inject(single_cardinality).unfold().as_(entity).addV(entity).as_("v").sideEffect(
        __.select(entity)
        .unfold()
        .as_("kv")
        .select("v")
        .property(__.select("kv").by(Column.keys), __.select("kv").by(Column.values))
    ).iterate()
    return g


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
