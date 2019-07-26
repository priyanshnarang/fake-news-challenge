# -*- coding: utf-8 -*-
from utils.dataset import DataSet
from utils.score import report_score, LABELS, score_submission
from utils.system import parse_params, check_version

if __name__ == "__main__":
    check_version()
    parse_params()

    #Load the training dataset and generate folds
    d = DataSet()



