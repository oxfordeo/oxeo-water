import operator
import re
from functools import reduce

import numpy as np
from treelib import Tree

from oxeo.water.models import Predictor

LABELS = {"clear": 0, "water": 1, "shadow": 2, "cloud": 3, "cirrus": 4, "snow": 5}

SEN2_TO_LANDSAT8 = {
    "B01": "B01",
    "B02": "B02",
    "B03": "B03",
    "B04": "B04",
    "B8A": "B05",
    "B10": "B09",
    "B11": "B06",
    "B12": "B07",
}


def B(arr, a):
    return arr.sel({"bands": [a]}).values


def S(arr, a, b):
    return B(arr, a) - B(arr, b)


def R(arr, a, b):
    return np.divide(B(arr, a), B(arr, b))


def thresh(a, thresh_value):
    return a < thresh_value


def eval_rule(arr, rule):
    if rule is None:
        return None
    if "leaf" in rule:
        return rule[rule.find("(") + 1 : rule.find(")")]

    bands = [b.strip() for b in rule[rule.find("(") + 1 : rule.find(")")].split(",")]

    thresh_value = eval(rule[rule.find("<") + 1 :])
    f = eval(rule[0])

    return thresh(f(arr, *bands), thresh_value).squeeze()


def extract_bands(rules):
    bands = set()
    for rule in rules:
        if rule is not None:
            bands.update(re.findall(r"(B\d[\d|A]|B\d)", rule))
    return list(bands)


def build_tree(arr, rules, constellation="sentinel-2"):
    """Build binary tree based on https://en.wikipedia.org/wiki/Binary_tree#Arrays

    Args:
        arr (xr.DataArray): The xarray DataArray
        rules (List[str]): list of rules in the form ['B(B01)<0.01', 'S(B02, B12)<1.0',..]
                        this list will be read as the array representation of a binary tree.

    Returns:
        [type]: the binary tree for the given array
    """
    bands = extract_bands(rules)
    arr = arr.sel({"bands": bands}).compute().astype(float)
    ftree = Tree()
    root_id = rules[0]
    root_data = eval_rule(arr, root_id)

    root_data = {"neg": ~root_data, "pos": root_data}
    ftree.create_node(identifier=root_id, data=root_data)
    for i in range(len(rules)):
        left_index = 2 * i + 1
        right_index = 2 * i + 2

        if right_index > len(rules) - 1:
            break
        parent_id = rules[i]
        left_id = rules[left_index]

        right_id = rules[right_index]

        left_data = eval_rule(arr, left_id)
        right_data = eval_rule(arr, right_id)

        parent_node = ftree.get_node(parent_id)
        if parent_node is not None:
            parent_data = parent_node.data

        if left_data is not None:

            if not isinstance(left_data, str):
                left_data = {
                    "neg": parent_data["neg"] & ~left_data,
                    "pos": parent_data["neg"] & left_data,
                }

            else:
                left_data = parent_data["neg"]

            ftree.create_node(identifier=left_id, parent=parent_id, data=left_data)

        if right_data is not None:
            if not isinstance(right_data, str):
                right_data = {
                    "neg": parent_data["pos"] & ~right_data,
                    "pos": parent_data["pos"] & right_data,
                }

            else:
                right_data = parent_data["pos"]
            ftree.create_node(identifier=right_id, parent=parent_id, data=right_data)
    return ftree


def predict(tree):
    mask = np.zeros(shape=tree.get_node(tree.root).data["pos"].shape, dtype=np.uint8)
    for l in LABELS.keys():
        label_leaves = [leave for leave in tree.leaves() if l in leave.tag]
        if len(label_leaves) > 0:
            label_mask = reduce(
                operator.__or__,
                (leaf.data for leaf in label_leaves),
            )

            mask[label_mask] = LABELS[l]
    return mask


def decision_tree_1(arr):
    rules = [
        "B(B03)<0.325",
        "B(B11)<0.267",
        "B(B8A)<0.166",
        "B(B07)<1.544",
        "B(B04)<0.674",
        "B(B10)<0.011",
        "B(B8A)<0.039",
        "leaf(snow)",
        "leaf(cloud)",
        "leaf(snow_2)",
        "leaf(cirrus)",
        "leaf(cirrus_2)",
        "leaf(clear)",
        "leaf(shadow)",
        "leaf(water)",
    ]
    return build_tree(arr, rules)


def decision_tree_landsat(arr):
    rules = [
        "B(B5)<0.181",
        "B(B1)<0.331",
        "B(B5)<0.051",
        "B(B6)<0.239",
        "B(B9)<0.012",
        "B(B7)<0.097",
        None,
        None,
        "B(B2)<0.711",
        "leaf(cirrus)",
        "B(B2)<0.271",
        "B(B9)<0.010",
        "B(B9)<0.011",
        None,
        None,
        None,
        None,
        "leaf(snow_2)",
        "leaf(cirrus_2)",
        None,
        None,
        "leaf(cloud_2)",
        "leaf(clear)",
        "leaf(shadow)",
        "leaf(clear_2)",
        "leaf(cirrus_3)",
        "leaf(shadow_2)",
        None,
        None,
        None,
        None,
    ]
    return build_tree(arr, rules, "landsat-8")


def decision_tree_2(arr):
    rules = [
        "B(B8A)<0.156",
        "B(B03)<0.333",
        "S(B06, B03)<-0.025",
        "R(B06, B11)<4.292",
        "R(B10,B02)<0.065",
        "S(B12,B09)<0.084",
        "S(B12,B09)<-0.016",
        "leaf(snow)",
        "leaf(cloud)",
        "leaf(cirrus)",
        "leaf(clear)",
        "leaf(clear_2)",
        "leaf(shadow)",
        "leaf(water)",
        "leaf(shadow_2)",
    ]
    return build_tree(arr, rules)


def decision_tree_3(arr):
    rules = [
        "B(B8A)<0.181",
        "B(B01)<0.331",
        "B(B8A)<0.051",
        "B(B11)<0.239",
        "B(B10)<0.012",
        "B(B12)<0.097",
        "B(B09)<0.010",
        "B(B05)<1.393",
        "B(B02)<0.711",
        "leaf(cirrus)",
        "B(B02)<0.271",
        "B(B10)<0.010",
        "B(B10)<0.011",
        "B(B03)<0.074",
        "B(B02)<0.073",
        "leaf(snow)",
        "leaf(cloud)",
        "leaf(snow_2)",
        "leaf(cirrus_2)",
        None,
        None,
        "leaf(cloud_2)",
        "leaf(clear)",
        "leaf(shadow)",
        "leaf(clear_2)",
        "leaf(cirrus_3)",
        "leaf(shadow_2)",
        "leaf(water)",
        "leaf(shadow_3)",
        "leaf(water_2)",
        "leaf(shadow_4)",
    ]
    return build_tree(arr, rules)


def decision_tree_4(arr):
    rules = [
        "B(B03)<0.319",
        "R(B05,B11)<4.330",
        "B(B8A)<0.166",
        "B(B03)<0.525",
        "S(B11,B10)<0.255",
        "R(B02,B10)<14.689",
        "S(B03,B07)<0.027",
        "leaf(snow)",
        "R(B01,B05)<1.184",
        "B(B01)<0.300",
        "S(B06,B07)<-0.016",
        "leaf(clear)",
        "R(B02,B09)<0.788",
        "S(B09,B11)<0.021",
        "S(B09, B11)<-0.097",
        None,
        None,
        "leaf(shadow)",
        "leaf(clear_2)",
        "leaf(cloud)",
        "leaf(clear_3)",
        "leaf(cirrus)",
        "leaf(cloud_2)",
        None,
        None,
        "leaf(cirrus_2)",
        "leaf(clear_4)",
        "leaf(shadow_2)",
        "leaf(water)",
        "leaf(shadow_3)",
        "leaf(clear_5)",
    ]
    return build_tree(arr, rules)


class HollsteinPredictor(Predictor):
    pass
