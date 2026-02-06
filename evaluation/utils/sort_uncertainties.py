def sort_uncertainties_image_level(uncertainties: dict):
    image_level_dict = {
        key: value["image_level"]["max_score"] for key, value in uncertainties.items()
    }
    image_level_dict = sorted(image_level_dict, key=image_level_dict.get, reverse=True)
    # print(image_level_dict)
    return image_level_dict


def sort_uncertainties_patch_level(uncertainties: dict):
    patch_level_dict = {
        key: value["patch_level"]["max_score"] for key, value in uncertainties.items()
    }
    patch_level_dict = sorted(patch_level_dict, key=patch_level_dict.get, reverse=True)
    # print(patch_level_dict)
    return patch_level_dict


def sort_uncertainties_threshold_level(uncertainties: dict):
    threshold_level_dict = {
        key: value["threshold"]["max_score"] for key, value in uncertainties.items()
    }
    threshold_level_dict = sorted(
        threshold_level_dict, key=threshold_level_dict.get, reverse=True
    )
    # print(threshold_level_dict)
    return threshold_level_dict


def sort_uncertainties_area_normalized(uncertainties: dict):
    area_level_dict = {
        key: value["area_normalized"]["max_score"]
        for key, value in uncertainties.items()
    }
    area_level_dict = sorted(area_level_dict, key=area_level_dict.get, reverse=True)
    return area_level_dict


def sort_uncertainties_border_normalized(uncertainties: dict):
    border_level_dict = {
        key: value["border_normalized"]["max_score"]
        for key, value in uncertainties.items()
    }
    border_level_dict = sorted(border_level_dict, key=border_level_dict.get, reverse=True)
    return border_level_dict


def sort_uncertainties(uncertainties: dict, level: str):
    if level == "image_level":
        return sort_uncertainties_image_level(uncertainties)
    elif level == "patch_level":
        return sort_uncertainties_patch_level(uncertainties)
    elif level == "threshold":
        return sort_uncertainties_threshold_level(uncertainties)
    elif level == "area_normalized":
        return sort_uncertainties_area_normalized(uncertainties)
    elif level == "border_normalized":
        return sort_uncertainties_border_normalized(uncertainties)
    else:
        raise Exception(f"Uncertainty level not known: {level}")
