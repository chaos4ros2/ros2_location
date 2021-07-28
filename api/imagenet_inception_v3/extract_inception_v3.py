import json
from typing import Dict, List

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras


def get_label(json_path: str = "./data/image_net_labels.json") -> List[str]:
    with open(json_path, "r") as f:
        labels = json.load(f)
    return labels


def load_hub_model() -> tf.keras.Model:
    # model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/classification/4")])
    # model = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")])
   

    # https://www.tensorflow.org/hub/api_docs/python/hub/load?hl=ja 
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" 
    model = hub.load(module_handle).signatures['default']
    
    return model


class InceptionV3Model(tf.keras.Model):
    def __init__(self, model: tf.keras.Model):
        super().__init__(self)
        self.model = model
        # self.labels = labels

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="image")])
    def serving_fn(self, input_img: str) -> tf.Tensor:
        def _base64_to_array(img):
            img = tf.io.decode_base64(img)
            img = tf.io.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            # 元の画像をリサイズしない
            # img = tf.image.resize(img, (299, 299))
            # img = tf.reshape(img, (299, 299, 3))
            return img

        img = tf.map_fn(_base64_to_array, input_img, dtype=tf.float32)
        # predictions = self.model(img)
        # https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb#scrollTo=kwGJV96WWBLH
        # load_img()、run_detector()内の処理を参照
        result = self.model(img)
        result = {key:value.numpy() for key,value in result.items()}

        print("Found %d objects." % len(result["detection_scores"]))

        def draw_bounding_box_on_image_(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
            """Adds a bounding box to an image."""
            draw = ImageDraw.Draw(image)
            im_width, im_height = image.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)
            draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                       (left, top)],
                      width=thickness,
                      fill=color)

            # If the total height of the display strings added to the top of the bounding
            # box exceeds the top of the image, stack the strings below the bounding box
            # instead of above.
            display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
            # Each display_str has a top and bottom margin of 0.05x.
            total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

            if top > total_display_str_height:
              text_bottom = top
            else:
              text_bottom = top + total_display_str_height
            # Reverse list and print from bottom to top.
            for display_str in display_str_list[::-1]:
              text_width, text_height = font.getsize(display_str)
              margin = np.ceil(0.05 * text_height)
              draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                              (left + text_width, text_bottom)],
                             fill=color)
              draw.text((left + margin, text_bottom - text_height - margin),
                        display_str,
                        fill="black",
                        font=font)
              text_bottom -= text_height - 2 * margin

        def draw_boxes_(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
            """Overlay labeled boxes on an image with formatted scores and label names."""
            colors = list(ImageColor.colormap.values())

            try:
              font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                                        25)
            except IOError:
              print("Font not found, using default font.")
              font = ImageFont.load_default()

            for i in range(min(boxes.shape[0], max_boxes)):
              if scores[i] >= min_score:
                ymin, xmin, ymax, xmax = tuple(boxes[i])
                display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                               int(100 * scores[i]))
                color = colors[hash(class_names[i]) % len(colors)]
                image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
                draw_bounding_box_on_image_(
                    image_pil,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                    color,
                    font,
                    display_str_list=[display_str])
                np.copyto(image, np.array(image_pil))
            return image

        image_with_boxes = draw_boxes_(
            img.numpy(), result["detection_boxes"],
            result["detection_class_entities"], result["detection_scores"])

        return image_with_boxes
        # def _convert_to_label(predictions):
        #     max_prob = tf.math.reduce_max(predictions)
        #     idx = tf.where(tf.equal(predictions, max_prob))
        #     label = tf.squeeze(tf.gather(self.labels, idx))
        #     return label

        # return tf.map_fn(_convert_to_label, predictions, dtype=tf.string)

    def save(self, export_path="./saved_model/inception_v3/"):
        signatures = {"serving_default": self.serving_fn}
        tf.keras.backend.set_learning_phase(0)
        tf.saved_model.save(self, export_path, signatures=signatures)


def main():
    # 物体検出の場合ラベルを別途に用意する必要はない
    # labels = get_label(json_path="./data/image_net_labels.json")
    inception_v3_hub_model = load_hub_model()
    inception_v3_model = InceptionV3Model(model=inception_v3_hub_model)
    version_number = 0
    inception_v3_model.save(export_path=f"./saved_model/inception_v3/{version_number}")


if __name__ == "__main__":
    main()
