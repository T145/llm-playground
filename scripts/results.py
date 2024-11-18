import os
import json

for subdir, dirs, files in os.walk('test_results/'):
    for f in files:
        fpath = os.path.join(subdir, f)

        if fpath.endswith('.json'):
            with open(fpath) as results:
                results = json.load(results)
                model_name = results["model_name"]
                results = results["results"]
                tests = [results[test] for test in ["tinyArc", "tinyGSM8k", "tinyHellaswag", "tinyMMLU", "tinyTruthfulQA", "tinyWinogrande"]]
                score = 0
                for test in tests:
                    for v in test.values():
                        if isinstance(v, float):
                            score += v
                print(model_name, score)
