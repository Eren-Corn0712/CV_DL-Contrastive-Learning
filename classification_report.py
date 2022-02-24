from sklearn.metrics import classification_report

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1]
target_names = ['class 0', 'class 1']
result = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

print(result)

def myprint(d):
    for k, v in d.items():
        if isinstance(v, dict):
            myprint(v)
        else:
            print("{0} : {1}".format(k, v))


myprint(result)