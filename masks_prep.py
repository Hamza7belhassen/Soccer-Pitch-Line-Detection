import cv2
import json
import os
import re
import numpy as np
from SoccerNet.Downloader import SoccerNetDownloader as SNdl


class SoccernetDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    @staticmethod
    def soccernet_downloader(dataset_dir="/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet/"):
        soccerNetDownloader = SNdl(LocalDirectory=dataset_dir)
        soccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])

    @staticmethod
    def show_image(image, size=(400, 600)):
        resized_image = cv2.resize(image, size)
        cv2.imshow("image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_read(self, image_path):
        image = cv2.imread(image_path)
        # resize_image = cv2.resize(image, self.size)
        return image

    def json_read(self, json_path):
        file = open(json_path)
        data = json.load(file)
        file.close()
        return data

    @staticmethod
    def generate_black_image(size):
        black_image = np.zeros(size, dtype=np.uint8)
        return black_image

    @staticmethod
    def draw_line(source_img, start, end):
        """
        :param source_img:
        :param start: (,)
        :param end: (,)
        :return:
        """
        img_with_line = np.copy(source_img)
        cv2.line(img_with_line, start, end, (255, 255, 255), thickness=2)
        return img_with_line

    @staticmethod
    def draw_point(source_img, point):
        img_with_point = np.copy(source_img)
        cv2.circle(img_with_point, point, radius=2, color=(200, 200, 200), thickness=2)
        return img_with_point

    def fit_circle(self, points_list):

        # Extract x and y coordinates of the points
        x = [point['x'] for point in points_list]
        y = [point['y'] for point in points_list]

        # Estimate the center of the circle using the mean of the points
        xc_initial = np.mean(x)
        yc_initial = np.mean(y)

        # Estimate the rayon of the circle using the maximum distance from the center to points
        r_initial = np.max(np.sqrt((np.array(x) - xc_initial) ** 2 + (np.array(y) - yc_initial) ** 2))

        return xc_initial, yc_initial, r_initial

    def angle_with_center(self, point, center):
        """Calculate the angle between the given point and the center of the circle."""
        return np.arctan2(point['y'] - center[1], point['x'] - center[0])

    def reorder_points(self, points, center):
        """Reorder the points to form a circle."""
        # Calculate the angle of each point with respect to the center
        angles = [self.angle_with_center(point, center) for point in points]
        # Sort the points based on their angles
        sorted_indices = np.argsort(angles)
        sorted_points = [points[i] for i in sorted_indices]

        return sorted_points

    def generate_output_image_m(self, size, data_specs):
        res_image = self.generate_black_image(size)
        for element_name, points_list in data_specs.items():
            if "Circle" in element_name:
                # Fit a circle to the points
                xc, yc, r = self.fit_circle(points_list)
                # Reorder the points to form a circle
                sorted_points = self.reorder_points(points_list, (xc, yc))
                # Draw the circle
                for i in range(len(sorted_points) - 1):
                    start_point = (int(sorted_points[i]['x'] * size[1]), int(sorted_points[i]['y'] * size[0]))
                    end_point = (int(sorted_points[i + 1]['x'] * size[1]), int(sorted_points[i + 1]['y'] * size[0]))
                    res_image = self.draw_line(res_image, start_point, end_point)
                # Draw the last segment to close the circle
                start_point = (int(sorted_points[-1]['x'] * size[1]), int(sorted_points[-1]['y'] * size[0]))
                end_point = (int(sorted_points[0]['x'] * size[1]), int(sorted_points[0]['y'] * size[0]))
                res_image = self.draw_line(res_image, start_point, end_point)
            else:
                # For lines, draw directly
                for i in range(len(points_list) - 1):
                    start_point = (int(points_list[i]['x'] * size[1]), int(points_list[i]['y'] * size[0]))
                    end_point = (int(points_list[i + 1]['x'] * size[1]), int(points_list[i + 1]['y'] * size[0]))
                    res_image = self.draw_line(res_image, start_point, end_point)
        return res_image

    def build_train_dataset(self):
        files_in_path = os.listdir(self.data_path + "/test/test/")
        images, json_files = {}, {}
        ctr = 0
        for f in files_in_path[:]:
            if re.search('jpg', f):
                img_path = self.data_path + "/test/test/" + f
                img = self.image_read(img_path)
                img_name = re.sub(".jpg", '', f)
                images[img_name] = img
            elif re.search('json', f) and not re.search('info', f):
                json_path = self.data_path + "/test/test/" + f
                my_json = self.json_read(json_path)
                json_name = re.sub(".json", '', f)
                json_files[json_name] = my_json
            ctr += 1
            if ctr % 100 == 0:
                print(f"{ctr}/{len(files_in_path)} process")
        return images, json_files

    def save_image(self, image, filename, save_directory):
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        cv2.imwrite(os.path.join(save_directory, filename + ".jpg"), image)




data_path = "/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet/calibration-2023"
output_folder = "/Desktop/train_cnn_f"
sn = SoccernetDataLoader(data_path)
images, json_files = sn.build_train_dataset()

for img_name, img in images.items():
    size = img.shape[:2]
    new_size = (540,960)
    bw = sn.generate_output_image_m(new_size, json_files[img_name])
    output_path = os.path.join(output_folder, f"{img_name}_output.jpg")
    sn.save_image(bw, output_path)



