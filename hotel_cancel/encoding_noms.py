from sklearn.preprocessing import LabelEncoder

def encoding_label(x):
    enc_classes = {}
    le = LabelEncoder()
    le.fit(x)
    label = le.transform(x)
    enc_classes[x.name] = le.classes_

    return label