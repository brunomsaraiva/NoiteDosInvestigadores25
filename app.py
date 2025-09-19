from flask import Flask, render_template, request, jsonify, session, send_file
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io
import base64
import os

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"


# Initialize session variables
@app.before_request
def init_session():
    if "phase1_score" not in session:
        session["phase1_score"] = 0
    if "phase2_score" not in session:
        session["phase2_score"] = 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/fase1")
def fase1():
    return render_template("fase1.html")


@app.route("/fase2")
def fase2():
    return render_template("fase2.html")


@app.route("/fase3")
def fase3():
    accuracy = session.get("phase2_accuracy", 0)
    progress = session.get("phase2_progress", 0)

    # Calculate training quality as multiplication of accuracy and progress (as percentages)
    training_quality = (accuracy * progress) / 100

    print(
        f"DEBUG: Phase 3 accessed - accuracy: {accuracy}, progress: {progress}, training_quality: {training_quality}"
    )

    return render_template(
        "fase3.html",
        phase2_accuracy=accuracy,
        phase2_progress=progress,
        phase2_training_quality=training_quality,
    )


def load_tif_image(image_path):
    """Load TIF image and convert to PIL Image"""
    if os.path.exists(image_path):
        # Load TIF image using PIL
        img = Image.open(image_path)

        # Ensure we always convert to RGB for consistent processing
        if img.mode != "RGB":
            if img.mode == "L":  # Grayscale
                img = img.convert("RGB")
            elif img.mode == "LA":  # Grayscale with alpha
                img = img.convert("RGB")
            elif img.mode == "P":  # Palette
                img = img.convert("RGB")
            elif img.mode == "RGBA":  # RGB with alpha
                img = img.convert("RGB")
            elif img.mode in ["I", "F"]:  # 32-bit integer or float
                # Normalize to 0-255 range and convert to RGB
                img = img.convert("L").convert("RGB")
            elif img.mode in ["I;16", "I;16B", "I;16L"]:  # 16-bit modes
                # Convert 16-bit to 8-bit by normalizing to 0-255
                img_array = np.array(img)
                # Normalize to 0-255 range
                min_val, max_val = img_array.min(), img_array.max()
                img_normalized = (
                    (img_array - min_val) * 255.0 / (max_val - min_val)
                ).astype(np.uint8)
                img = Image.fromarray(img_normalized, mode="L").convert("RGB")
            else:
                # Force conversion to RGB for any other mode
                img = img.convert("RGB")

        return img
    else:
        # Create a dummy image if file doesn't exist
        return Image.new("RGB", (512, 512), color="gray")


def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


@app.route("/api/process_image", methods=["POST"])
def process_image():
    data = request.json

    # Get slider values
    sharpness = data.get("sharpness", 50)
    contrast = data.get("contrast", 50)
    brightness = data.get("brightness", 50)
    noise_filter = data.get("noise_filter", 50)
    zoom = data.get("zoom", 100)

    # Ideal values for perfect image
    ideal_values = {
        "sharpness": 70,
        "contrast": 60,
        "brightness": 50,
        "noise_filter": 40,
        "zoom": 100,
    }

    # Load original clean TIF image
    img_path = "static/images/cell_fluor_1.tif"
    img = load_tif_image(img_path)

    # Calculate deviation from ideal for each parameter
    sharpness_dev = abs(sharpness - ideal_values["sharpness"]) / 100.0
    contrast_dev = abs(contrast - ideal_values["contrast"]) / 100.0
    brightness_dev = abs(brightness - ideal_values["brightness"]) / 100.0
    noise_dev = abs(noise_filter - ideal_values["noise_filter"]) / 100.0
    zoom_dev = abs(zoom - ideal_values["zoom"]) / 100.0

    # Apply distortions based on deviation from ideal
    # The further from ideal, the more distorted the image becomes

    # Apply blur based on sharpness deviation
    if sharpness_dev > 0.05:  # 5% tolerance
        blur_radius = sharpness_dev * 4  # Scale blur amount
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Apply brightness distortion
    if brightness_dev > 0.05:
        enhancer = ImageEnhance.Brightness(img)
        # If brightness slider is wrong, make image too dark or too bright
        if brightness < ideal_values["brightness"]:
            brightness_factor = 0.4 + (brightness / 100.0) * 0.6
        else:
            brightness_factor = (
                1.0 + (brightness - ideal_values["brightness"]) / 100.0 * 0.8
            )
        img = enhancer.enhance(brightness_factor)

    # Apply contrast distortion
    if contrast_dev > 0.05:
        enhancer = ImageEnhance.Contrast(img)
        # If contrast slider is wrong, reduce contrast
        if contrast < ideal_values["contrast"]:
            contrast_factor = 0.3 + (contrast / 100.0) * 0.7
        else:
            contrast_factor = (
                1.0 + (contrast - ideal_values["contrast"]) / 100.0 * 0.5
            )
        img = enhancer.enhance(contrast_factor)

    # Add noise based on noise filter deviation
    if noise_dev > 0.05:
        img_array = np.array(img)
        noise_amount = noise_dev * 25  # Scale noise amount
        noise = np.random.normal(0, noise_amount, img_array.shape)
        noise = noise.astype(np.uint8)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255)
        img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array)

    # Apply zoom
    if zoom_dev > 0.05:
        zoom_factor = zoom / 100.0
        width, height = img.size
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop to original size if zoomed
        if zoom_factor != 1.0:
            if zoom_factor > 1.0:
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                img = img.crop((left, top, left + width, top + height))
            else:
                # For zoom out, pad the image
                new_img = Image.new("RGB", (width, height), color="black")
                paste_x = (width - new_width) // 2
                paste_y = (height - new_height) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img

    # Calculate score
    current_values = {
        "sharpness": sharpness,
        "contrast": contrast,
        "brightness": brightness,
        "noise_filter": noise_filter,
        "zoom": zoom,
    }

    score = calculate_score(current_values, ideal_values)

    return jsonify({"image": image_to_base64(img), "score": score})


@app.route("/api/save_phase1_score", methods=["POST"])
def save_phase1_score():
    score = request.json.get("score", 0)
    session["phase1_score"] = score
    return jsonify({"success": True})


@app.route("/api/save_phase2_score", methods=["POST"])
def save_phase2_score():
    accuracy = request.json.get("accuracy", 0)
    progress = request.json.get("progress", 0)

    session["phase2_accuracy"] = accuracy
    session["phase2_progress"] = progress

    print(
        f"DEBUG: Saved to session - accuracy: {accuracy}, progress: {progress}"
    )
    return jsonify({"success": True})


@app.route("/api/test_session")
def test_session():
    """Test endpoint to check session state"""
    session_score = session.get("phase2_score", "NOT_SET")
    return jsonify(
        {
            "phase2_score": session_score,
            "session_id": session.get("_id", "NO_ID"),
        }
    )


@app.route("/api/set_test_score/<int:accuracy>/<int:progress>")
def set_test_score(accuracy, progress):
    """Test endpoint to manually set accuracy and progress scores"""
    session["phase2_accuracy"] = accuracy
    session["phase2_progress"] = progress
    return jsonify(
        {"success": True, "accuracy": accuracy, "progress": progress}
    )


@app.route("/api/get_original_image")
def get_original_image():
    img_path = "static/images/cell_fluor_1.tif"
    img = load_tif_image(img_path)

    # Convert to PNG for web display
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.route("/api/get_original_image2")
def get_original_image2():
    """Get the second image for phase 3"""
    img_path = "static/images/cell_fluor_2.tif"
    img = load_tif_image(img_path)

    # Convert to PNG for web display
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.route("/api/get_distorted_image")
def get_distorted_image():
    """Get distorted version of the original image for phase 1"""
    img_path = "static/images/cell_fluor_1.tif"
    img = load_tif_image(img_path)

    # Apply initial distortions
    img = img.filter(ImageFilter.GaussianBlur(radius=2))

    # Reduce brightness and contrast
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.7)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.6)

    # Add noise
    img_array = np.array(img)
    noise = np.random.normal(0, 20, img_array.shape).astype(np.uint8)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(
        np.uint8
    )
    img = Image.fromarray(img_array)

    # Convert to PNG for web display
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.route("/api/get_distorted_image2")
def get_distorted_image2():
    """Get distorted version of the second image for phase 3"""
    img_path = "static/images/cell_fluor_2.tif"
    img = load_tif_image(img_path)

    # Apply distortions similar to phase 1
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.7)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(0.6)

    # Convert to PNG for web display
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype="image/png")


@app.route("/api/get_cell_labels")
def get_cell_labels():
    """Get cell label positions from the TIF label file for Phase 2"""
    label_path = "static/images/cell_labels_1.tif"
    try:
        # Load label image
        label_img = Image.open(label_path)
        label_array = np.array(label_img)

        # Find connected components (cell regions)
        # Convert to binary if needed
        if len(label_array.shape) > 2:
            label_array = label_array[:, :, 0]  # Take first channel

        # Find unique labels (excluding background)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background

        cell_positions = []
        # Type casting to handle numpy shape type issues
        height = int(label_array.shape[0])  # type: ignore
        width = int(label_array.shape[1])  # type: ignore

        for label_val in unique_labels:
            # Find center of mass for each labeled region
            y_coords, x_coords = np.where(label_array == label_val)
            if len(x_coords) > 0:
                center_x = np.mean(x_coords) / width
                center_y = np.mean(y_coords) / height
                cell_positions.append(
                    {"x": float(center_x), "y": float(center_y)}
                )

        return jsonify(
            {
                "cell_positions": cell_positions,
                "total_cells": len(unique_labels),
            }
        )

    except Exception:
        # Fallback to hardcoded positions if file reading fails
        fallback_positions = [
            {"x": 0.2, "y": 0.3},
            {"x": 0.4, "y": 0.2},
            {"x": 0.6, "y": 0.4},
            {"x": 0.3, "y": 0.6},
            {"x": 0.7, "y": 0.7},
            {"x": 0.1, "y": 0.8},
            {"x": 0.8, "y": 0.2},
            {"x": 0.5, "y": 0.8},
        ]
        return jsonify(
            {
                "cell_positions": fallback_positions,
                "total_cells": len(fallback_positions),
            }
        )


@app.route("/api/get_cell_labels_overlay")
def get_cell_labels_overlay():
    """Get cell label overlay image for Phase 2 perfect visualization"""
    try:
        # Load the original fluorescence image
        fluor_path = "static/images/cell_fluor_1.tif"
        fluor_img = load_tif_image(fluor_path)

        # Load the label image
        label_path = "static/images/cell_labels_1.tif"
        label_img = Image.open(label_path)
        label_array = np.array(label_img)

        # Convert to binary if needed
        if len(label_array.shape) > 2:
            label_array = label_array[:, :, 0]

        # Create overlay image
        overlay_img = fluor_img.copy()
        overlay_array = np.array(overlay_img)

        # Find unique labels (excluding background)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels > 0]

        # Create colored overlay for each cell
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
        ]

        for i, label_val in enumerate(unique_labels):
            # Get the mask for this cell
            cell_mask = label_array == label_val
            color = colors[i % len(colors)]

            # Find cell boundaries using simple edge detection
            # Dilate the mask and subtract original to get boundaries
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(
                cell_mask.astype(np.uint8), kernel, iterations=1
            )
            boundaries = (dilated - cell_mask.astype(np.uint8)) > 0

            # Apply color to boundaries with some transparency
            for channel in range(3):
                orig_val = overlay_array[boundaries, channel]
                new_val = orig_val * 0.7 + color[channel] * 0.3
                overlay_array[boundaries, channel] = new_val.astype(np.uint8)

        # Convert back to PIL Image
        overlay_img = Image.fromarray(overlay_array)

        # Convert to PNG for web display
        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    except Exception:
        # Return original image if overlay fails
        return get_original_image()


@app.route("/api/get_all_cell_masks")
def get_all_cell_masks():
    """Get all cell masks with different colors for Phase 2"""
    label_path = "static/images/cell_labels_1.tif"
    try:
        # Load label image
        label_img = Image.open(label_path)
        label_array = np.array(label_img)

        # Convert to binary if needed
        if len(label_array.shape) > 2:
            label_array = label_array[:, :, 0]

        # Create transparent overlay
        height = int(label_array.shape[0])  # type: ignore
        width = int(label_array.shape[1])  # type: ignore

        # Create RGBA overlay (with transparency)
        overlay_array = np.zeros((height, width, 4), dtype=np.uint8)

        # Find unique labels (excluding background)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels > 0]

        # Define colors for different cells
        colors = [
            (255, 0, 0, 100),  # Red with transparency
            (0, 255, 0, 100),  # Green with transparency
            (0, 0, 255, 100),  # Blue with transparency
            (255, 255, 0, 100),  # Yellow with transparency
            (255, 0, 255, 100),  # Magenta with transparency
            (0, 255, 255, 100),  # Cyan with transparency
            (255, 128, 0, 100),  # Orange with transparency
            (128, 0, 255, 100),  # Purple with transparency
        ]

        for i, label_val in enumerate(unique_labels):
            # Get the mask for this cell
            cell_mask = label_array == label_val
            color = colors[i % len(colors)]

            # Find cell boundaries
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(
                cell_mask.astype(np.uint8), kernel, iterations=2
            )
            boundaries = (dilated - cell_mask.astype(np.uint8)) > 0

            # Fill boundaries with solid color
            overlay_array[boundaries] = [color[0], color[1], color[2], 255]

            # Fill interior with semi-transparent color
            overlay_array[cell_mask] = color

        # Convert to PIL Image with transparency
        overlay_img = Image.fromarray(overlay_array, "RGBA")

        # Convert to PNG for web display
        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    except Exception:
        # Return empty transparent image if fails
        empty_img = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        buffer = io.BytesIO()
        empty_img.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")


@app.route("/api/validate_cell_click", methods=["POST"])
def validate_cell_click():
    """Validate if a click is inside a cell and return the cell mask"""
    data = request.json
    if data is None:
        return jsonify({"is_valid": False, "error": "No data provided"})

    click_x = data.get("x", 0)  # Normalized coordinates (0-1)
    click_y = data.get("y", 0)

    label_path = "static/images/cell_labels_1.tif"
    try:
        # Load label image
        label_img = Image.open(label_path)
        label_array = np.array(label_img)

        # Convert to binary if needed
        if len(label_array.shape) > 2:
            label_array = label_array[:, :, 0]

        # Type casting to handle numpy shape type issues
        height = int(label_array.shape[0])  # type: ignore
        width = int(label_array.shape[1])  # type: ignore

        # Convert normalized coordinates to pixel coordinates
        pixel_x = int(click_x * width)
        pixel_y = int(click_y * height)

        # Check bounds
        if 0 <= pixel_x < width and 0 <= pixel_y < height:
            label_value = label_array[pixel_y, pixel_x]

            if label_value > 0:  # Clicked inside a cell
                # Get the mask for this specific cell
                cell_mask = (label_array == label_value).astype(np.uint8)

                # Create colored overlay image
                fluor_path = "static/images/cell_fluor_1.tif"
                fluor_img = load_tif_image(fluor_path)
                overlay_array = np.array(fluor_img)

                # Create a bright overlay for the selected cell
                cell_color = (0, 255, 0)  # Green

                # Find cell boundaries
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(cell_mask, kernel, iterations=2)
                boundaries = (dilated - cell_mask) > 0

                # Fill the cell area with semi-transparent color
                for channel in range(3):
                    overlay_array[cell_mask == 1, channel] = (
                        overlay_array[cell_mask == 1, channel] * 0.7
                        + cell_color[channel] * 0.3
                    ).astype(np.uint8)

                # Make boundaries more prominent
                for channel in range(3):
                    overlay_array[boundaries, channel] = cell_color[channel]

                # Convert to base64
                overlay_img = Image.fromarray(overlay_array)
                overlay_base64 = image_to_base64(overlay_img)

                return jsonify(
                    {
                        "is_valid": True,
                        "label_value": int(label_value),
                        "overlay_image": overlay_base64,
                    }
                )
            else:
                return jsonify({"is_valid": False, "label_value": 0})
        else:
            return jsonify({"is_valid": False, "label_value": 0})

    except Exception as e:
        return jsonify({"is_valid": False, "error": str(e)})


@app.route("/api/get_cell_labels2")
def get_cell_labels2():
    """Get cell label positions from the second TIF label file for phase 3"""
    label_path = "static/images/cell_labels_2.tif"
    try:
        # Load label image
        label_img = Image.open(label_path)
        label_array = np.array(label_img)

        # Find connected components (cell regions)
        if len(label_array.shape) > 2:
            label_array = label_array[:, :, 0]  # Take first channel

        # Find unique labels (excluding background)
        unique_labels = np.unique(label_array)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background

        cell_positions = []
        height = int(label_array.shape[0])  # type: ignore
        width = int(label_array.shape[1])  # type: ignore

        for label_val in unique_labels:
            # Find center of mass for each labeled region
            y_coords, x_coords = np.where(label_array == label_val)
            if len(x_coords) > 0:
                center_x = np.mean(x_coords) / width
                center_y = np.mean(y_coords) / height
                cell_positions.append(
                    {"x": float(center_x), "y": float(center_y)}
                )

        print(
            f"DEBUG: Loaded {len(cell_positions)} cell positions from {label_path}"
        )
        return jsonify({"cell_positions": cell_positions})

    except Exception:
        # Fallback to hardcoded positions if file reading fails
        fallback_positions = [
            {"x": 0.15, "y": 0.25},
            {"x": 0.35, "y": 0.15},
            {"x": 0.55, "y": 0.35},
            {"x": 0.25, "y": 0.55},
            {"x": 0.65, "y": 0.65},
            {"x": 0.05, "y": 0.75},
            {"x": 0.75, "y": 0.15},
            {"x": 0.45, "y": 0.75},
        ]
        return jsonify({"cell_positions": fallback_positions})


def calculate_score(current, ideal):
    total_deviation = 0
    for param in ideal:
        deviation = abs(current[param] - ideal[param]) / 100.0
        total_deviation += deviation

    # Convert to percentage (lower deviation = higher score)
    max_deviation = len(ideal) * 1.0  # Maximum possible deviation
    score = max(0, 100 * (1 - total_deviation / max_deviation))
    return round(score, 1)


if __name__ == "__main__":
    # Create images directory if it doesn't exist
    os.makedirs("static/images", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5001)
