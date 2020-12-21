from exp_utils.model_utils import load_cifar_data, load_properties_data
from matplotlib import pyplot as plt
import torch


# This function acts aims to find adversarial examples for each property in the file specified. It acts as a container
# for the function which attacks each property in turn by calling this function for each property.

def pgd_attack_properties(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_epochs, num_trials,
                          subset=None):
    # Load all the required data for the images which were correctly verified by the model
    images, true_labels, image_indices, model = load_cifar_data(model_name)

    # Update the images and true_labels lists and load the lists of the test labels and epsilons such that they
    # correspond to the properties appearing in the properties dataframe which were correctly verified
    images, true_labels, test_labels, epsilons = load_properties_data(properties_filename, images, true_labels,
                                                                      image_indices)

    # If the subset of indices was specified for the purpose of reducing the time complexity, drop the elements of
    # images, true_labels, test_labels and epsilons not indicated in the subset indices
    original_length = len(images)
    if subset is not None:
        for i in range(original_length - 1, -1, -1):
            if i not in subset:
                images.pop(i)
                true_labels.pop(i)
                test_labels.pop(i)
                epsilons.pop(i)

    # Now attack each property in turn by calling the appropriate function
    num_properties_still_verified = 0  # counter of properties which are still verified after the PGD attack
    for i in range(len(images)):
        successful_attack_flag = pgd_attack_property(model, images[i], true_labels[i], test_labels[i], epsilons[i],
                                                     epsilon_factor, pgd_learning_rate, num_epochs, num_trials)

        # If the attack was unsuccessful, increase the counter
        if not successful_attack_flag:
            num_properties_still_verified += 1

    # Calculate the verification accuracy for the properties in the file provided after all the PGD attacks
    verification_accuracy = 100.0 * num_properties_still_verified / len(images)

    return verification_accuracy


# This function performs the PGD attack on the specified property characterised by its image, true label, test label and
# epsilon value

def pgd_attack_property(model, image, true_label, test_label, epsilon, epsilon_factor, pgd_learning_rate, num_epochs,
                        num_trials):

    # First, perturb the image randomly within the allowed bounds and perform a PGD attack
    lower_bound = torch.add(-epsilon * epsilon_factor, image)
    upper_bound = torch.add(epsilon * epsilon_factor, image)
    perturbation = torch.add(-epsilon * epsilon_factor,
                             2 * epsilon * epsilon_factor * torch.rand(image.size()))
    perturbed_image = torch.add(image, perturbation).clone().detach().requires_grad_(True)
    successful_attack_flag, heuristics_dict = gradient_ascent(model, perturbed_image, lower_bound, upper_bound,
                                                              true_label, test_label, pgd_learning_rate, num_epochs)

    # If the attack was successful, the procedure can be terminated and True can be returned
    if successful_attack_flag:
        return True

    return


# This function performs Gradient Ascent on the specified property given the bounds on the input

def gradient_ascent(model, perturbed_image, lower_bound, upper_bound, true_label, test_label, pgd_learning_rate,
                    num_epochs):
    # Initialise the relevant optimiser and the heuristics dictionary to be used if the PGD attack is unsuccessful
    optimizer = torch.optim.Adam([perturbed_image], lr=pgd_learning_rate)
    heuristics_dict = {}  # TODO

    # Perform Gradient Ascent for a specified number of epochs
    for epoch in range(num_epochs):
        logits = model(perturbed_image)[0]
        loss = -logit_difference_loss(logits, test_label, true_label)  # '-' sign since gradient ascent is performed

        # If the difference between the logit of the test class and the logit of the true class is positive,
        # then the PGD attack was successful and gradient ascent can be stopped
        if -loss > 0:
            return True, None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clip the values of the perturbed image so that they are within the allowed perturbation magnitude
        # This operation isn't related to optimisation, hence it is wrapped with torch.no_grad()
        with torch.no_grad():
            perturbed_image[:] = torch.max(torch.min(perturbed_image, upper_bound), lower_bound)

    # If the flag has not been set yet but the perturbation resulted in the model predicting the test class instead of
    # the true one during the last epoch, return the True successful attack flag
    if torch.max(model(perturbed_image)[0], 0)[1].item() == test_label:
        return True, None

    # If the Gradient Ascent didn't lead to the changed prediction, then output the heuristics gathered during the
    # Gradient Ascent and output them as well

    return False, heuristics_dict


def logit_difference_loss(logits, test_class, true_class):
    true_class_logit = logits[true_class]
    test_class_logit = logits[test_class]
    return test_class_logit - true_class_logit


pgd_attack_properties('base_easy.pkl', 'cifar_base_kw', 1, 0.01, 100, 0)
