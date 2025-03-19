# Read the file with cp949 encoding


for i in range(1, 12):
    with open(f"{i}.txt", "r", encoding="cp949") as file:
        content = file.read()

    # Modify content (example: append a new line)
    content += "\nNew line added!"

    # Save back with cp949 encoding
    with open(f"{i}_ve.txt", "w", encoding="utf-8") as file:
        file.write(content)

    print("File saved successfully with cp949 encoding.")
