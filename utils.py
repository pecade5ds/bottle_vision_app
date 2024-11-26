def filter_and_count(data, threshold=0.5, class_var="class"):
    filtered_data = [item for item in data if item["confidence"] >= threshold]
    result = {}
    for item in filtered_data:
        class_name = item[class_var]
        result[class_name] = result.get(class_name, 0) + 1
    return result
