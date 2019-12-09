from conlleval import return_report
import os
output_predict_file = os.path.join('output_my', "label_test.txt")
eval_result = return_report(output_predict_file)
print(eval_result)