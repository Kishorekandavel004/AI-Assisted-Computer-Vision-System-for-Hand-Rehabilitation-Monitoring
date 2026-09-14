import csv
import os
from datetime import datetime


DATA_FILE = "data/session_data.csv"


def save_session(
    patient_id,
    exercise,
    target_angle,
    max_rom,
    min_rom,
    repetitions,
    correct_repetitions,
    incorrect_repetitions
):

    os.makedirs(
        "data",
        exist_ok=True
    )

    if repetitions > 0:

        accuracy = (
            correct_repetitions /
            repetitions
        ) * 100

    else:

        accuracy = 0

    file_exists = os.path.isfile(
        DATA_FILE
    )

    with open(
        DATA_FILE,
        mode="a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([

                "Patient_ID",

                "Date",

                "Time",

                "Exercise",

                "Target_Angle",

                "Min_Angle",

                "Max_Angle",

                "ROM_Range",

                "Repetitions",

                "Correct_Repetitions",

                "Incorrect_Repetitions",

                "Accuracy"
            ])

        rom_range = (
            max_rom -
            min_rom
        )

        now = datetime.now()

        writer.writerow([

            patient_id,

            now.strftime(
                "%Y-%m-%d"
            ),

            now.strftime(
                "%H:%M:%S"
            ),

            exercise,

            target_angle,

            round(
                min_rom,
                2
            ),

            round(
                max_rom,
                2
            ),

            round(
                rom_range,
                2
            ),

            repetitions,

            correct_repetitions,

            incorrect_repetitions,

            round(
                accuracy,
                2
            )
        ])

    print(
        "Session saved successfully."
    )