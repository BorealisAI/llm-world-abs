# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


def pad_dict(list_of_dict):
    all_k = []
    new_lod = []
    for lod in list_of_dict:
        all_k += list(lod.keys())

    all_k = list(set(all_k))

    for lod in list_of_dict:
        new_lod.append(
            {k: (lod[k] if k in lod.keys() else None) for k in all_k}
        )

    return new_lod


def calc_label_wght(labels, label_size):
    wght = [0. for _ in range(label_size)]
    total = len(labels)
    for y in range(label_size):
        wght[y] = len([x for x in labels if x==y]) / (total+1e-8)
    return wght


def get_maxN_of_nobj_for_qopt(predicates_data):
    maxN = 0

    for data in predicates_data:

        maxN = max(
            maxN,
            max(
                [
                    x["label"] for x in list(data["qopt"].values())
                ]
            )
            )
    return maxN


def summ_acc(df):
    obj_cn = [x.replace("_label", "") for x in df.columns if x.startswith("obj") and x.endswith("label")]

    def obj_corr_ctrl(row, reverse):
        if reverse:
            obj_interest = [x for x in obj_cn
                            if not np.isnan(row[x+'_label'])
                            and x not in row['obj_to_ctrl']
                            and row[x+'_label']!=-100]
        else:
            obj_interest = [x for x in row['obj_to_ctrl'] if not np.isnan(row[x+'_label']) and row[x+'_label']!=-100]

        ret = [1. if row[x+"_label"]==row[x+"_pred"] else 0. for x in obj_interest]
        if len(ret) == 0:
            return None
        return np.mean(ret)

    def obj_corr_hidden_info(row, reverse):
        if reverse:
            obj_interest = [x for x in obj_cn
                        if not np.isnan(row[x+'_label']) and "obj_loc" not in row['hidden_info'] and row[x+'_label']!=-100]
        else:
            obj_interest = [x for x in obj_cn
                        if not np.isnan(row[x+'_label']) and "obj_loc" in row['hidden_info'] and row[x+'_label']!=-100]
        
        ret = [1. if row[x+"_label"]==row[x+"_pred"] else 0. for x in obj_interest]
        if len(ret) == 0:
            return None
        return np.mean(ret)
    
    def obj_corr_overall(row):
        obj_interest = [x for x in obj_cn if not np.isnan(row[x+'_label']) and row[x+'_label']!=-100]
        ret = [1. if row[x+"_label"]==row[x+"_pred"] else 0. for x in obj_interest]
        if len(ret) == 0:
            return None
        return np.mean(ret)

    obj_loc_acc = {}

    corr = df.apply(lambda row: obj_corr_ctrl(row, False), axis=1)
    corr = [x for x in corr if x is not None and not np.isnan(x)]
    obj_loc_acc["ctrl"] = np.mean(corr)

    corr = df.apply(lambda row: obj_corr_ctrl(row, True), axis=1)
    corr = [x for x in corr if x is not None and not np.isnan(x)]
    obj_loc_acc["not-ctrl"] = np.mean(corr)

    corr = df.apply(lambda row: obj_corr_hidden_info(row, False), axis=1)
    corr = [x for x in corr if x is not None and not np.isnan(x)]
    obj_loc_acc["hidden"] = np.mean(corr)

    corr = df.apply(lambda row: obj_corr_hidden_info(row, True), axis=1)
    corr = [x for x in corr if x is not None and not np.isnan(x)]
    obj_loc_acc["not-hidden"] = np.mean(corr)

    corr = df[df.modifier].apply(lambda row: obj_corr_overall(row), axis=1)
    corr = [x for x in corr if x is not None and not np.isnan(x)]
    obj_loc_acc["modifier"] = np.mean(corr)

    corr = df[~df.modifier].apply(lambda row: obj_corr_overall(row), axis=1)
    corr = [x for x in corr if x is not None and not np.isnan(x)]
    obj_loc_acc["not-modifier"] = np.mean(corr)

    aloc_acc = {}
    aloc_acc["changed"] = df[df.hidden_info.apply(lambda x: "agent_loc" in x)].apply(
        lambda row: 1. if row['agent_loc_pred']==row['agent_loc_label'] else 0.,
        axis=1
    ).mean()
    aloc_acc["keep"] = df[df.hidden_info.apply(lambda x: "agent_loc" not in x)].apply(
        lambda row: 1. if row['agent_loc_pred']==row['agent_loc_label'] else 0.,
        axis=1
    ).mean()

    aloc_acc['modifier'] = df[df.modifier].apply(
        lambda row: 1. if row['agent_loc_pred']==row['agent_loc_label'] else 0.,
        axis=1).mean()

    aloc_acc['not-modifier'] = df[~df.modifier].apply(
        lambda row: 1. if row['agent_loc_pred']==row['agent_loc_label'] else 0.,
        axis=1).mean()

    box_order_acc = {}
    box_order_acc['modifier'] = df[df.modifier].apply(
        lambda row: 1. if row['boxes_order_pred']==row['boxes_order_label'] else 0.,
        axis=1).mean()

    box_order_acc['not-modifier'] = df[~df.modifier].apply(
        lambda row: 1. if row['boxes_order_pred']==row['boxes_order_label'] else 0.,
        axis=1).mean()

    print('obj_loc_acc')
    for k, v in obj_loc_acc.items():
        print(k, v)

    print("aloc_acc")
    for k, v in aloc_acc.items():
        print(k, v)

    print("box_order_acc")
    for k, v in box_order_acc.items():
        print(k, v)