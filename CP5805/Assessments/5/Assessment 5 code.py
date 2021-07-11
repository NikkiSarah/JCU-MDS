# Assessment 5: Advanced Data Analysis
# Python code
# 13848336 Nikki Fitzherbert

import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    data_frame = pd.DataFrame()
    valid_menu_options = ["1", "2", "3", "4"]
    display_main_menu()

    menu_choice = int(get_menu_choice(valid_menu_options))
    while menu_choice != 4:
        if menu_choice == 1:
            data = load_data()
            data_frame = data_frame.append(data)
        elif len(data_frame) > 0:
            if menu_choice == 2:
                display_statistical_report(data_frame)
            elif menu_choice == 3:
                generate_plot(data_frame)
        else:
            print("What are you doing? You need to first load a CSV file.\n")
        display_main_menu()
        menu_choice = int(get_menu_choice(valid_menu_options))
    print("Goodbye. You have exited the program.")


def display_main_menu():
    print("""Welcome to the DataFrame Statistician!
Please choose from the following options:
  1 - Load data from a CSV file
  2 - Analyse
  3 - Visualise
  4 - Quit""")


def get_menu_choice(valid_menu_options):
    menu_choice = input("Please make a selection by entering a number from 1 to 4: ")
    while menu_choice not in valid_menu_options:
        print("Invalid entry. Please try again.")
        menu_choice = input("Please make a selection by entering a number from 1 to 4: ")
    return menu_choice


def load_data():
    file = input("Please enter the name of your CSV file including the extension: ")
    valid_file = validate_file_pointer(file)
    data = pd.read_csv(valid_file)
    return data


def validate_file_pointer(file):
    valid_file = False
    while not valid_file:
        try:
            open(file, 'r')
            valid_file = True
        except FileNotFoundError:
            print("There is no such file. Please try again.")
            file = input("Please enter the name of your CSV file including the extension: ")
    while os.stat(file).st_size < 5:
        print("No data found in that CSV file! Please try again.")
        file = input("Please enter the name of your CSV file including the extension: ")
    return file


def display_variable_menu(data_frame):
    number_of_variables = len(data_frame.columns)
    data_frame_names = list(data_frame.columns.values)

    print("Which variable do you want to analyse?")
    for integer in range(1, number_of_variables + 1):
        variable_name = data_frame_names[integer - 1]
        print(" ", integer, "-", variable_name)
    return number_of_variables, data_frame_names


def get_variable_choice(number_of_variables):
    integer_input = False

    variable_choice = input("Please make a selection by entering a number from 1 to {}: ".format(number_of_variables))
    while not integer_input:
        try:
            variable_choice = int(variable_choice)
            while int(variable_choice) not in range(1, number_of_variables + 1):
                print("Invalid entry. Please try again.")
                variable_choice = input(
                    "Please make a selection by entering a number from 1 to {}: ".format(number_of_variables))
            integer_input = True
        except ValueError:
            print("Invalid entry. Please try again.")
            variable_choice = input(
                "Please make a selection by entering a number from 1 to {}: ".format(number_of_variables))
    return variable_choice


def calculate_statistics(data_frame_names, variable_choice, data_frame):
    chosen_variable = data_frame[data_frame_names[int(variable_choice) - 1]]

    number_of_values = chosen_variable.count()
    mean = chosen_variable.mean()
    standard_deviation = chosen_variable.std()
    standard_error = chosen_variable.sem()

    statistics = [number_of_values, mean, standard_deviation, standard_error]
    return statistics


def display_statistical_report(data_frame):
    number_of_variables, data_frame_names = display_variable_menu(data_frame)
    variable_choice = get_variable_choice(number_of_variables)
    statistics = calculate_statistics(data_frame_names, variable_choice, data_frame)

    print(data_frame_names[int(variable_choice) - 1])
    print("----------")
    print("{:<20}".format("Number of values (n)" + ": " + str(statistics[0])))
    print("{:<20}".format("Mean" + "{:>18}".format(": ") + "{:<20.2f}".format(statistics[1])))
    print("{:<20}".format("Standard deviation" + "{:>4}".format(": ") + "{:<20.2f}".format(statistics[2])))
    print("{:<20}".format("Standard error" + "{:>8}".format(": ") + "{:<20.2f}".format(statistics[3])))
    print("----------\n")


def display_plotting_menu():
    print("""You have the option of the following plots:
  1 – Line chart
  2 – Bar chart
  3 – Box plot
... and you have the option of the following plot layouts:
  1 – Single plot
  2 – Subplots""")


def get_plotting_choice():
    valid_plotting_options = ["11", "12", "21", "22", "31", "32"]

    plotting_choice = input(
        "Please make a selection by entering both numbers together. For example, if you wanted to plot a single line "
        "chart you would enter 11: ")
    while plotting_choice not in valid_plotting_options:
        print("Invalid entry. Please try again.")
        plotting_choice = input("Please make a selection by entering both numbers together: ")
    return plotting_choice


def generate_plot(data_frame):
    display_plotting_menu()
    plotting_choice = int(get_plotting_choice())

    if plotting_choice == 11:
        data_frame.plot()
        plt.xlabel("index")
        plt.ylabel("value")
    elif plotting_choice == 12:
        data_frame.plot(subplots=True)
        plt.xlabel("index")
        plt.ylabel("value")
    elif plotting_choice == 21:
        data_frame.plot.bar(stacked=True)
        plt.xlabel("index")
        plt.ylabel("value")
    elif plotting_choice == 22:
        data_frame.plot.bar(subplots=True, legend=False)
        plt.xlabel("index")
        plt.ylabel("value")
    elif plotting_choice == 31:
        data_frame.plot.box()
        plt.xlabel("variable")
    elif plotting_choice == 32:
        data_frame.plot.box(subplots=True)

    plt.show()


main()