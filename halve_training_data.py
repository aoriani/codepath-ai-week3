def halve_file(input_file, output_file):
    line_count = 0
    with open(input_file, 'r') as the_input, open(output_file, 'w') as output:
        for line in the_input:
            if line_count % 2 == 0:
                output.write(line)
            line_count = line_count + 1


if __name__ == "__main__":
    halve_file("trainingsets/iteration5/facts_training_5.jsonl",
               "trainingsets/iteration6/facts_training_6.jsonl")
    halve_file("trainingsets/iteration5/facts_validation_5.jsonl",
               "trainingsets/iteration6/facts_validation_6.jsonl")
