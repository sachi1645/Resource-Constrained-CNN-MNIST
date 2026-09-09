from src.models.model_utils import count_parameters


def check_parameter_limit(model, max_parameters=100000):
    """
    Check whether a model satisfies the maximum
    trainable-parameter constraint.
    """

    trainable_parameters = count_parameters(model)

    return {
        "trainable_parameters": trainable_parameters,
        "max_parameters": max_parameters,
        "passed": trainable_parameters <= max_parameters,
    }


def print_parameter_report(model, model_name, max_parameters=None):
    """
    Print a simple parameter report for a model.
    """

    trainable_parameters = count_parameters(model)

    print("\n" + "=" * 50)
    print(f"{model_name} Parameter Report")
    print("=" * 50)

    print(f"Trainable parameters : {trainable_parameters:,}")

    if max_parameters is not None:
        print(f"Maximum allowed     : {max_parameters:,}")

        if trainable_parameters <= max_parameters:
            print("Constraint status   : PASSED")
        else:
            print("Constraint status   : FAILED")

    print("=" * 50)