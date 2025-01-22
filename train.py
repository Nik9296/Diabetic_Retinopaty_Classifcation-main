# import os
# import shutil
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Paths
# data_dir = 'data'
# train_dir = 'train'
# val_dir = 'val'
# test_dir = 'test'


# # Create the directory structure for training, validation, and test
# def create_class_directories(base_dir, class_names):
#     for class_name in class_names:
#         os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

# class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']
# create_class_directories(train_dir, class_names)
# create_class_directories(val_dir, class_names)
# create_class_directories(test_dir, class_names)

# # Load the CSV file
# df = pd.read_csv("train.csv")

# # Print column names for debugging
# print("CSV columns:", df.columns)

# # Ensure the 'diagnosis' column exists
# if 'diagnosis' not in df.columns or 'id_code' not in df.columns:
#     raise ValueError("The CSV file is missing required columns.")

# # Mapping dictionary
# diagnosis_dict = {
#     0: 'No_DR',
#     1: 'Mild',
#     2: 'Moderate',
#     3: 'Severe',
#     4: 'Proliferate_DR',
# }

# # Add type column based on diagnosis
# df['type'] = df['diagnosis'].map(diagnosis_dict.get)

# # Split data
# train, test_and_val = train_test_split(df, train_size=0.7, stratify=df['type'])
# test, val = train_test_split(test_and_val, test_size=0.5, stratify=test_and_val['type'])

# # Function to move images
# def move_images(df_subset, src_dir, dest_dir):
#     for index, row in df_subset.iterrows():
#         diagnosis = row['type']
#         id_code = row['id_code'] + ".png"
#         srcfile = os.path.join(src_dir, diagnosis, id_code)
#         dstfile = os.path.join(dest_dir, diagnosis, id_code)

#         # Debugging output
#         print(f"Source: {srcfile}")
#         print(f"Destination: {dstfile}")

#         if os.path.exists(srcfile):
#             shutil.copy(srcfile, dstfile)
#             print(f"Copied {id_code} to {dstfile}")
#         else:
#             print(f"File not found: {srcfile}")

# # Move images to respective directories
# move_images(train, data_dir, train_dir)
# move_images(val, data_dir, val_dir)
# move_images(test, data_dir, test_dir)

# print("Directory setup complete.")



import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define directories
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

# Comment out or remove the directory setup code if directories are already set up
# def setup_directory_structure(src_dir, dst_dir, df):
#     if not os.path.exists(dst_dir):
#         os.makedirs(dst_dir)
#     for index, row in df.iterrows():
#         diagnosis = row['type']
#         binary_diagnosis = row['binary_type']
#         id_code = row['id_code'] + ".png"
#         srcfile = os.path.join(src_dir, diagnosis, id_code)
#         dstfile = os.path.join(dst_dir, binary_diagnosis)
#         os.makedirs(dstfile, exist_ok=True)
#         shutil.copy(srcfile, dstfile)
#
# # Assuming 'df' is the DataFrame containing the dataset information
# setup_directory_structure("data", train_dir, df)
# setup_directory_structure("data", val_dir, df)
# setup_directory_structure("data", test_dir, df)

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_batches = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_batches = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_batches = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='softmax')  # 5 classes in the output
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=val_batches
)

# Save the model
model.save('retina_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_batches)
print(f'Test accuracy: {test_acc}')

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
