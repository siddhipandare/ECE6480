import matplotlib.pyplot as plt
from torch import flatten
import cv2
import numpy as np
from skimage import io, transform
import torch
from torchvision import transforms, utils
from skimage.transform import resize
from torch import nn
import torch.nn.functional as F
import tkinter as tk
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
global original_image, predicted_image, preprocessed_image

# Image Preprocessing Helper functions
def gamma_correction(img, gamma=1.0):
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')
    return gamma_corrected

def select_gamma(img):
    av = np.average(img)
    gamma = av / 30
    return gamma

def apply_clahe(img, limit):
    clahe = cv2.createCLAHE(clipLimit=limit)
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = clahe.apply(img)
    return img

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def preprocess_img(img, num_sections):
    sections = []
    gammas = []
    gamma_corrected_xray = img.copy()

    for i in range(num_sections):
        if i == 0:
            sections.append(gamma_corrected_xray[:int(gamma_corrected_xray.shape[0] / num_sections), :])
        elif i == num_sections - 1:
            sections.append(gamma_corrected_xray[int(gamma_corrected_xray.shape[0] * (i) / num_sections + 1):, :])
        else:
            sections.append(gamma_corrected_xray[int(gamma_corrected_xray.shape[0] * (i) / num_sections):int(
                gamma_corrected_xray.shape[0] * (i + 1) / num_sections), :])

        gammas.append(select_gamma(sections[i]))
        sections[i] = gamma_correction(sections[i], gammas[i])

        sections[i] = 255 * normalize(sections[i])

    for i in range(num_sections):
        gamma_weighted = 255 * (gammas[i] / max(gammas))

        if i == 0:
            gamma_corrected_xray[:int(gamma_corrected_xray.shape[0] / num_sections), :] = sections[
                i]  # 255*normalize(((1/gamma_weighted)*img[:int(gamma_corrected_xray.shape[0]/num_sections), :] + gamma_weighted*sections[i])/2)
        elif i == num_sections - 1:
            gamma_corrected_xray[int(gamma_corrected_xray.shape[0] * (i) / num_sections + 1):, :] = sections[
                i]  # 255*normalize(((1/gamma_weighted)*img[int(gamma_corrected_xray.shape[0]*(i)/num_sections+1):, :] +gamma_weighted*sections[i])/2)
        else:
            gamma_corrected_xray[int(gamma_corrected_xray.shape[0] * (i) / num_sections):int(
                gamma_corrected_xray.shape[0] * (i + 1) / num_sections), :] = sections[
                i]  # 255*normalize(((1/gamma_weighted)*img[int(gamma_corrected_xray.shape[0]*(i)/num_sections):int(gamma_corrected_xray.shape[0]*(i+1)/num_sections), :] + gamma_weighted*sections[i])/2)
    return gamma_corrected_xray


# Classifier for the dataset.
class Spine4PointClassifier(nn.Module):

    def __init__(self, classes):
        super(Spine4PointClassifier, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), padding=3)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.Conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.Conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(5, 5), padding=2)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.FC1 = nn.Linear(in_features=65536, out_features=512)
        self.FC2 = nn.Linear(in_features=512, out_features=512)
        self.FC3 = nn.Linear(in_features=512, out_features=512)
        self.FC4 = nn.Linear(in_features=512, out_features=classes)

    outputs = list()

    def forward(self, x):
        output = F.relu(self.Conv1(x))
        output = self.MaxPool1(output)
        output = F.relu(self.Conv2(output))
        output = self.MaxPool2(output)
        output = F.relu(self.Conv3(output))
        output = self.MaxPool3(output)
        output = F.relu(self.Conv4(output))
        output = self.MaxPool4(output)
        output = torch.flatten(output, 1)
        output = F.relu(self.FC1(output))
        output = F.relu(self.FC2(output))
        output = F.relu(self.FC3(output))
        output = F.relu(self.FC4(output))

        return output


class FourPointModel():

    def __init__(self, modelFilePath_fourpoint, modelFilePath_centroids, height, width):
        self.device = torch.device('cpu')
        self.model = Spine4PointClassifier(classes=68 * 2).to(self.device)
        self.model.load_state_dict(torch.load(modelFilePath_fourpoint))
        self.num_sections = 100
        self.originalWidth = 0
        self.originalHeight = 0
        self.width = width
        self.height = height
        self.preprocessed_image = 0
        self.image = 0

        self.image_transforms = transforms.Compose([transforms.Resize((256, 128)), transforms.ToTensor()])
        self.centroidModel = SpineCentroidClassifier()
        self.centroidModel.load_state_dict(torch.load(modelFilePath_centroids))

    def __getImage__(self, image_path):
        image = io.imread(image_path)

        return image

    def __predict__(self, image_path):

        # preprocess image and rescale inputs and outputs
        self.preprocessed_image = self.__getPreprocessedImg__(image_path)
        self.preprocessed_image = resize(self.preprocessed_image, (self.height, self.width), anti_aliasing=False)

        self.preprocessed_image = np.reshape(self.preprocessed_image,
                                             (1, 1, self.preprocessed_image.shape[0], self.preprocessed_image.shape[1]))
        # Convert to Torch Tensors
        self.preprocessed_image = torch.from_numpy(np.float32(self.preprocessed_image))

        # Make Prediction
        predicted_labels = self.model(self.preprocessed_image)

        predicted_labels = predicted_labels.cpu().detach().numpy()[0].reshape((68, 2))

        return predicted_labels

    def __getPreprocessedImg__(self, image_path):
        originalImg = self.__getImage__(image_path)
        self.originalWidth = originalImg.shape[1]
        self.originalHeight = originalImg.shape[0]
        return preprocess_img(originalImg, self.num_sections)

    def __centroid_localisation__(self, image_path):
        self.centroidModel.eval()
        image = resize(self.__getPreprocessedImg__(image_path), (self.height, self.width), anti_aliasing=False)

        image = Image.fromarray(image)
        image = self.image_transforms((image)).float()
        image = image.unsqueeze(0)
        results = self.centroidModel(image)
        landmarks = results[0]
        landmarks = landmarks.cpu().detach().numpy().copy()

        points = list()
        index = 0
        for i in range(0, int(len(landmarks) / 2)):
            x_point = landmarks[index] * self.width

            y_point = landmarks[index + 1] * self.height

            points.append([x_point, y_point])
            index += 2

        return points

    def __getPredictedImg__(self, image_path):

        image = resize(self.__getPreprocessedImg__(image_path), (self.height, self.width), anti_aliasing=False) * 255

        predicted_labels = self.__predict__(image_path)

        points = self.__centroid_localisation__(image_path)

        s = 0
        while s < predicted_labels.shape[0]:
            x = int(predicted_labels[s][0] * self.width)
            y = int(predicted_labels[s][1] * self.height)

            i = 0
            j = 0

            for i in range(2):
                for j in range(2):
                    image[y + i][x + j] = 255.0
            s = s + 1

        print(points)

        return image, points

    def centroids2angle(self, X):

        # defining models
        input_shape = (34,)
        model4 = Sequential()
        model4.add(Dense(26, input_shape=input_shape, activation='relu'))
        model4.add(Dense(13, activation='relu'))
        model4.add(Dense(2, activation='linear'))
        optimizer4 = Adam(learning_rate=0.00001, beta_1=0.85, beta_2=0.99, epsilon=1e-10, )
        model4.compile(loss='mse', optimizer=optimizer4, metrics=['mae'])

        model5 = Sequential()
        model5.add(Dense(26, input_shape=input_shape, activation='relu'))
        model5.add(Dense(13, activation='relu'))
        model5.add(Dense(2, activation='linear'))
        optimizer5 = Adam(learning_rate=0.00001, beta_1=0.85, beta_2=0.99, epsilon=1e-10, )
        model5.compile(loss='mse', optimizer=optimizer5, metrics=['mae'])

        # loading weights
        model4.load_weights('arch3m1.h5')
        model5.load_weights('arch3m2.h5')

        # prediction
        Y_pred1 = model4.predict(X)
        Y_pred2 = model5.predict(X)
        Y_pred = [Y_pred1[0][0], (Y_pred1[0][1] + Y_pred2[0][0]) / 2, Y_pred2[0][1]]

        return (Y_pred)

class SpineCentroidClassifier(nn.Module):

    def __init__(self, classes=34):
        super(SpineCentroidClassifier, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (16,1,256,128) (batch_size, channels, height, width)

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3)
        # (16,32,256,128)
        self.relu1 = nn.ReLU()
        # (16,32,256,128)
        self.MaxPool1 = nn.MaxPool2d(kernel_size=2)
        # (16,32,128,64)

        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # (16,64,128,64)
        self.relu2 = nn.ReLU()
        # (16,64,128,64)
        self.MaxPool2 = nn.MaxPool2d(kernel_size=2)
        # (16,64,64,32)

        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # (16,128,64,32)
        self.relu3 = nn.ReLU()
        # (16,128,64,32)
        self.MaxPool3 = nn.MaxPool2d(kernel_size=2)
        # (16,128,32,16)

        self.Conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        # (16,256,32,16)
        self.relu4 = nn.ReLU()
        # (16,256,32,16)
        self.MaxPool4 = nn.MaxPool2d(kernel_size=2)
        # (16,256,16,8)

        self.FC1 = nn.Linear(in_features=256 * 16 * 8, out_features=512)
        self.relu5 = nn.ReLU()
        self.FC2 = nn.Linear(in_features=512, out_features=512)
        self.relu6 = nn.ReLU()
        self.FC3 = nn.Linear(in_features=512, out_features=classes)

    def forward(self, input):
        output = self.Conv1(input)
        output = self.relu1(output)
        output = self.MaxPool1(output)

        output = self.Conv2(output)
        output = self.relu2(output)
        output = self.MaxPool2(output)

        output = self.Conv3(output)
        output = self.relu3(output)
        output = self.MaxPool3(output)

        output = self.Conv4(output)
        output = self.relu4(output)
        output = self.MaxPool4(output)

        output = flatten(output, 1)

        output = self.FC1(output)
        output = self.relu5(output)
        output = self.FC2(output)
        output = self.relu6(output)
        output = self.FC3(output)

        return output


height = 256
width = 128
fpmodel = FourPointModel(r"best_outcorrection_train_validation_sunday1msererun2.tch", r"100_epochs.pth", height, width)

# Create GUI
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"





class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("X-ray Cobb angle calculation for patients with scoliosis")
        self.geometry(f"{1440}x{600}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=0) # column is width
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure(0, weight=0) # row is height
        self.grid_rowconfigure((1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Cobb angle calculation", font=customtkinter.CTkFont(size=30, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self.sidebar_frame, width=250)
        self.textbox.grid(row=3, column=0, padx=(20,20), pady=(20,20), sticky="nsew")

        # Create button
        self.Button = tk.Button(self.sidebar_frame, text='Upload File  (.jpg,.jpeg)', width=20, command=lambda:self.upload_file())
        self.Button.grid(row=4, column=0)

        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.textbox.insert("0.0", "Welcome to the automated Cobb angle calculation software\n\n" +
                            "Please click on the button below to upload an x-ray image\n\n" +
                            "Preprocessed image, predicted image and calculated Cobb angle will be displayed to the right\n")

    def upload_file(self):
        global original_image, predicted_image, preprocessed_image, trial
        f_types = [('Jpeg Files', '*.jpeg'), ('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)  # path of file

        original_img = resize(fpmodel.__getImage__(filename), (256*2, 128*2), anti_aliasing=False) * 255  # original image
        image, points = fpmodel.__getPredictedImg__(filename)
        scale_w = fpmodel.originalWidth / image.shape[1]
        scale_h = fpmodel.originalHeight / image.shape[0]


        points = np.array(points)
        points2predict = points.copy()
        points2predict[:, 0] *= scale_h
        points2predict[:, 1] *= scale_w
        points_ca = points2predict.flatten()
        points_ca = points_ca.tolist()
        points_ca = [points_ca]
        print(points_ca)
        three_cobb_angles = fpmodel.centroids2angle(points_ca)
        print(three_cobb_angles)
        plt.clf()
        plt.scatter(points[:, 0], points[:, 1], marker="o", color="red", s=20)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig('image.jpeg', bbox_inches='tight')
        predicted_img = resize(fpmodel.__getImage__("image.jpeg"), (256 * 2, 128 * 2),
                               anti_aliasing=False)  # predicted image
        preprocessed_img = resize(fpmodel.__getPreprocessedImg__(filename), (256*2, 128*2), anti_aliasing=False) * 255   # preprocessed image

        original_image = ImageTk.PhotoImage(image=Image.fromarray(original_img), format='PPM')  # original image
        predicted_image = ImageTk.PhotoImage(image=Image.fromarray((predicted_img * 255).astype(np.uint8)), format='PPM')  # predicted image
        preprocessed_image = ImageTk.PhotoImage(image=Image.fromarray(preprocessed_img),
                                                format='PPM')  # preprocessed image

        # Display Uploaded Image, Preprocessed Image, Predicted Image
        original_canvas = tk.Canvas(self, width=original_img.shape[1], height=original_img.shape[0])
        original_canvas.grid(row=1, column=1)
        original_canvas.create_image(0, 0, anchor='nw', image=original_image)

        preprocessed_canvas = tk.Canvas(self, width=preprocessed_img.shape[1], height=preprocessed_img.shape[0])
        preprocessed_canvas.grid(row=1, column=2)
        preprocessed_canvas.create_image(0, 0, anchor='nw', image=preprocessed_image)

        predicted_canvas = tk.Canvas(self, width=predicted_img.shape[1], height=predicted_img.shape[0])
        predicted_canvas.grid(row=1, column=3)
        predicted_canvas.create_image(0, 0, anchor='nw', image=predicted_image)

        # Create labels for image display
        Label_original = customtkinter.CTkLabel(self, text="Uploaded Image",
                                                     font=customtkinter.CTkFont(size=20, weight="bold"))
        Label_original.grid(row=0, column=1, pady=(20, 10), sticky="nsew")
        Label_preprocessed = customtkinter.CTkLabel(self, text="Preprocessed Image",
                                                         font=customtkinter.CTkFont(size=20, weight="bold"))
        Label_preprocessed.grid(row=0, column=2, pady=(20, 10), sticky="nsew")
        Label_predicted = customtkinter.CTkLabel(self, text="Predicted Image",
                                                      font=customtkinter.CTkFont(size=20, weight="bold"))
        Label_predicted.grid(row=0, column=3, pady=(20, 10), sticky="nsew")

        # Display Calculated Cobb Angle & Label
        Label_Cobb_Angle = customtkinter.CTkLabel(self, text="Calculated Cobb Angle:", font=customtkinter.CTkFont(size=20, weight="bold"))
        Label_Cobb_Angle.grid(row=0, column=4, pady=(20, 10), sticky="nsew")
        Textbox = customtkinter.CTkTextbox(self, width=250)
        Textbox.grid(row=1, column=4, padx=(20, 20), pady=(20, 20), sticky="nsew")
        cobb_angle_1 = str(round(three_cobb_angles[0], 1))
        cobb_angle_2 = str(round(three_cobb_angles[1], 1))
        cobb_angle_3 = str(round(three_cobb_angles[2], 1))
        Textbox.insert("0.0", "Three estimated Cobb angles are:\n\n" + cobb_angle_1 + ", " + cobb_angle_2 + ", " + cobb_angle_3)


    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")

    def on_closing(self):
        self.quit()


if __name__ == "__main__":

    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
