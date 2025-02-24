"""
This file is used to manually match low to high level concepts. They're saved to mappings.json.
"""
import os
import json


def read_low_level(file_path):
    """
    Parses the attributes file for low level concepts.
    """
    concepts = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            concepts.append(parts[1])
    return concepts


def read_high_level(file_path):
    """
    Reads body parts from a file and returns a list, adds general as an option.
    """
    high_level = ['general']  # Adding 'general' as a category
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            high_level.append(parts[-1])
    return high_level


def user_map(low_level, high_level):
    """
    Asks the user to match low level to high level concepts.
    """
    matches = {}
    seen = {}
    total = 0

    print("Please match each attribute to a body part.")
    for concept in low_level:
        # See if we've done this base concept
        base_concept = concept.split("::")[0]
        if base_concept in seen:
            matches[concept] = high_level[seen[base_concept]]
            continue

        print(concept)
        choice = int(input("Part ID: "))
        if choice == -1:
            return matches
        matches[concept] = high_level[choice]
        seen[base_concept] = choice
        total += 1
        print("------------")
    print(f"Total assignments: {total}")
    return matches


def save_as_json(matches, filename):
    with open(filename, 'w') as json_file:
        json.dump(matches, json_file, indent=4)
    print(f"Matches have been saved to '{filename}'.")


def main():
    cub_path = os.getenv("CUB_PATH")
    low_path = os.path.join(cub_path, "attributes/attributes.txt")
    high_path = os.path.join(cub_path, "parts/parts.txt")

    # There's the weird issue of having left and right for some concepts
    # Just only put left for now. We'll rename these to "legs" or something
    # If either are visible, the entire higher level concept will be represented
    low_level = read_low_level(low_path)
    high_level = read_high_level(high_path)
    mappings = user_map(low_level, high_level)
    save_as_json(mappings, "mappings.json")


if __name__ == "__main__":
    main()
