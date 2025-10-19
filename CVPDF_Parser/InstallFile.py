import kagglehub

# Download latest version
path = kagglehub.dataset_download("mehyarmlaweh/ner-annotated-cvs")

print("Path to dataset files:", path)