## This script will calculate summary statistics of phylogenetic trees

### Based on from DOI:10.1371/journal.pcbi.1005416

"""
Notation Description
max_H Sum of the branch lengths between the root and its farthest leaf
min_H Sum of the branch lengths between the root and its closest leaf
a_BL_mean Mean length of all branches
a_BL_median Median length of all branches
a_BL_var Variance of the lengths of all branches
e_BL_mean Mean length of external branches
e_BL_median Median length of external branches
e_BL_var Variance of the lengths of external branches
i_BL_mean_[k]· Piecewise mean length of internal branches
i_BL_median_[k]· Piecewise median length of internal branches
i_BL_var_[k]· Piecewise variance of the lengths of internal branches
ie_BL_mean_[k]  Ratio of the piecewise mean length of internal branches over the mean length of
external branches
ie_BL_median_[k]  Ratio of the piecewise median length of internal branches over the median length of
external branches
ie_BL_var_[k]  Ratio of the piecewise variance of the lengths of internal branches over the variance
of the lengths of external branches
colless Sum for each internal node of the absolute difference between the number of leaves on
the left side and the number of leaves on the right side [47]
sackin Sum for each leaf of the number of internal nodes between the leaf and the root [48]
WD_ratio Ratio of the maximal width (W) over the maximal depth (D), where the depth of a node
characterizes the number of branches that lies between it and the root, and the width wd
of a tree at a depth level d is the number of nodes that have the same depth d [49]
Δw Maximal difference in width Dw  maxD􀀀 1
d0 jwd 􀀀 wd1j [49]
max_ladder Maximal number of internal nodes in a ladder which is a chain of connected internal
nodes each linked to a single leaf, divided by the number of leaves [49]
IL_nodes Proportion of internal nodes In Ladders [49]
staircaseness_1 Proportion of imbalanced internal nodes that have different numbers of leaves between
the left and the right side [49]
staircaseness_2 Mean ratio of the minimal number of leaves on a side over the maximal number of leaves
on a side, for each internal node [49, 50]
"""