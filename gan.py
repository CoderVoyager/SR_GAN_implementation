import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# Keras imports
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, PReLU, BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from keras.applications import VGG19

# Create directory structure
def setup_directories():
    directories = ['data', 'data/lr_images', 'data/hr_images', 'results', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directory structure created successfully.")

# Generate dataset from a collection of images
def generate_dataset(source_dir, n_images=100, hr_size=128, lr_size=32):
    """
    Create a dataset of low-resolution and high-resolution image pairs
    
    Parameters:
    source_dir - Directory containing source images to process
    n_images - Number of image pairs to generate
    hr_size - Size of high-resolution images (hr_size x hr_size)
    lr_size - Size of low-resolution images (lr_size x lr_size)
    """
    print(f"Generating dataset: {n_images} LR-HR image pairs...")
    
    # Get list of image files from source directory
    try:
        all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' not found.")
        print("Using random generated images instead...")
        # Generate random images if source directory doesn't exist
        generate_random_dataset(n_images, hr_size, lr_size)
        return
    
    if len(all_images) == 0:
        print("No images found in source directory. Using random generated images instead...")
        generate_random_dataset(n_images, hr_size, lr_size)
        return
    
    # Limit to the requested number of images
    num_to_process = min(n_images, len(all_images))
    selected_images = all_images[:num_to_process]
    
    for i, img_file in enumerate(tqdm(selected_images)):
        try:
            # Read image
            img_path = os.path.join(source_dir, img_file)
            img = cv2.imread(img_path)
            
            # Skip if image loading failed
            if img is None:
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to high-resolution size
            hr_img = cv2.resize(img, (hr_size, hr_size), interpolation=cv2.INTER_CUBIC)
            
            # Resize to low-resolution size
            lr_img = cv2.resize(img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
            
            # Save images
            cv2.imwrite(f"data/hr_images/{i:04d}.png", cv2.cvtColor(hr_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"data/lr_images/{i:04d}.png", cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    print(f"Generated {num_to_process} LR-HR image pairs successfully.")

# Generate random images for dataset if no source images are available
def generate_random_dataset(n_images=100, hr_size=128, lr_size=32):
    """
    Create a dataset of random low-resolution and high-resolution image pairs
    
    Parameters:
    n_images - Number of image pairs to generate
    hr_size - Size of high-resolution images (hr_size x hr_size)
    lr_size - Size of low-resolution images (lr_size x lr_size)
    """
    print(f"Generating {n_images} random LR-HR image pairs...")
    
    for i in tqdm(range(n_images)):
        # Create a random high-resolution image with patterns
        hr_img = np.zeros((hr_size, hr_size, 3), dtype=np.uint8)
        
        # Add some color gradients
        for c in range(3):
            channel = np.zeros((hr_size, hr_size))
            num_gradients = random.randint(1, 5)
            
            for _ in range(num_gradients):
                x1, y1 = random.randint(0, hr_size-1), random.randint(0, hr_size-1)
                x2, y2 = random.randint(0, hr_size-1), random.randint(0, hr_size-1)
                
                for x in range(hr_size):
                    for y in range(hr_size):
                        dist = np.sqrt((x - x1)**2 + (y - y1)**2) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        channel[y, x] += np.maximum(0, 1 - dist) * random.uniform(0.5, 1.0)
                        
            # Normalize and set the channel
            channel = (channel / channel.max() * 255).astype(np.uint8)
            hr_img[:, :, c] = channel
        
        # Add some shapes
        for _ in range(random.randint(0, 3)):
            shape_type = random.randint(0, 2)
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            
            if shape_type == 0:  # Rectangle
                x1, y1 = random.randint(0, hr_size-20), random.randint(0, hr_size-20)
                w, h = random.randint(10, min(hr_size-x1, 50)), random.randint(10, min(hr_size-y1, 50))
                cv2.rectangle(hr_img, (x1, y1), (x1+w, y1+h), color, -1)
                
            elif shape_type == 1:  # Circle
                x, y = random.randint(20, hr_size-20), random.randint(20, hr_size-20)
                radius = random.randint(5, 20)
                cv2.circle(hr_img, (x, y), radius, color, -1)
                
            else:  # Line
                x1, y1 = random.randint(0, hr_size-1), random.randint(0, hr_size-1)
                x2, y2 = random.randint(0, hr_size-1), random.randint(0, hr_size-1)
                thickness = random.randint(1, 5)
                cv2.line(hr_img, (x1, y1), (x2, y2), color, thickness)
        
        # Create the low-resolution version
        lr_img = cv2.resize(hr_img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
        
        # Save images
        cv2.imwrite(f"data/hr_images/{i:04d}.png", hr_img)
        cv2.imwrite(f"data/lr_images/{i:04d}.png", lr_img)
    
    print(f"Generated {n_images} random LR-HR image pairs successfully.")

# Generate a sample test image
def generate_test_image(filename='test_image', size=256):
    """Generate a sample test image for super-resolution"""
    # Create a new image with gradient and shapes
    test_img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add gradient background
    for y in range(size):
        for x in range(size):
            test_img[y, x, 0] = int(255 * (x / size))  # Red channel
            test_img[y, x, 1] = int(255 * (y / size))  # Green channel
            test_img[y, x, 2] = int(255 * ((size-x) / size) * ((size-y) / size))  # Blue channel
    
    # Add some shapes
    # Circle
    cv2.circle(test_img, (size//4, size//4), size//8, (255, 0, 0), -1)
    
    # Rectangle
    cv2.rectangle(test_img, (size//2, size//2), (3*size//4, 3*size//4), (0, 255, 0), -1)
    
    # Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(test_img, 'SRGAN', (size//8, 7*size//8), font, 1, (255, 255, 255), 2)
    
    # Save high-res version
    cv2.imwrite(f"data/{filename}_hr.jpg", test_img)
    
    # Create and save low-res version (32x32)
    lr_size = 32
    lr_img = cv2.resize(test_img, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"data/{filename}_lr.jpg", lr_img)
    
    print(f"Test image '{filename}' generated successfully.")
    return lr_img, test_img

# Define residual block for the generator
def res_block(ip):
    res_model = Conv2D(64, (3,3), padding="same")(ip)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    res_model = PReLU(shared_axes=[1,2])(res_model)
    
    res_model = Conv2D(64, (3,3), padding="same")(res_model)
    res_model = BatchNormalization(momentum=0.5)(res_model)
    
    return add([ip, res_model])

# Define upscale block for the generator
def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D(size=2)(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)
    
    return up_model

# Create generator model
def create_gen(gen_ip, num_res_block):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)
    
    temp = layers
    
    for i in range(num_res_block):
        layers = res_block(layers)
    
    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers, temp])
    
    layers = upscale_block(layers)
    layers = upscale_block(layers)
    
    op = Conv2D(3, (9,9), padding="same")(layers)
    
    return Model(inputs=gen_ip, outputs=op)

# Define discriminator block
def discriminator_block(ip, filters, strides=1, bn=True):
    disc_model = Conv2D(filters, (3,3), strides=strides, padding="same")(ip)
    
    if bn:
        disc_model = BatchNormalization(momentum=0.8)(disc_model)
    
    disc_model = LeakyReLU(alpha=0.2)(disc_model)
    
    return disc_model

# Create discriminator model
def create_disc(disc_ip):
    df = 64
    
    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)
    
    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    
    return Model(disc_ip, validity)

# Build VGG19 model for perceptual loss
def build_vgg(hr_shape):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
    
    return Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

# Create combined GAN model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    
    gen_features = vgg(gen_img)
    
    disc_model.trainable = False
    validity = disc_model(gen_img)
    
    return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])

# Train SRGAN model
def train_srgan(epochs=20, batch_size=1, sample_interval=50, save_interval=10):
    """
    Train the SRGAN model
    
    Parameters:
    epochs - Number of training epochs
    batch_size - Batch size for training
    sample_interval - Interval for sampling and preview
    save_interval - Interval for saving the model
    """
    # Load training images
    print("Loading training data...")
    try:
        lr_list = os.listdir("data/lr_images")
        hr_list = os.listdir("data/hr_images")
        
        if len(lr_list) == 0 or len(hr_list) == 0:
            raise FileNotFoundError("No images found in data directories")
    except FileNotFoundError:
        print("Dataset not found. Generating sample dataset...")
        generate_random_dataset(n_images=100)
        lr_list = os.listdir("data/lr_images")
        hr_list = os.listdir("data/hr_images")
    
    # Sort the lists to ensure corresponding pairs
    lr_list.sort()
    hr_list.sort()
    
    lr_images = []
    hr_images = []
    
    print("Processing image data...")
    for img_lr, img_hr in tqdm(zip(lr_list, hr_list), total=len(lr_list)):
        # Load images
        img_lr = cv2.imread(os.path.join("data/lr_images", img_lr))
        img_hr = cv2.imread(os.path.join("data/hr_images", img_hr))
        
        if img_lr is None or img_hr is None:
            continue
        
        # Convert BGR to RGB
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        
        lr_images.append(img_lr)
        hr_images.append(img_hr)
    
    # Convert to numpy arrays
    lr_images = np.array(lr_images)
    hr_images = np.array(hr_images)
    
    # Display sample image pair
    image_number = random.randint(0, len(lr_images)-1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Low Resolution')
    plt.imshow(lr_images[image_number])
    plt.subplot(122)
    plt.title('High Resolution')
    plt.imshow(hr_images[image_number])
    plt.savefig("results/sample_pair.png")
    plt.close()
    
    # Normalize images
    lr_images = lr_images / 255.0
    hr_images = hr_images / 255.0
    
    # Split into train and test sets
    lr_train, lr_test, hr_train, hr_test = train_test_split(
        lr_images, hr_images, test_size=0.2, random_state=42)
    
    # Get image shapes
    hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
    lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])
    
    # Print dataset information
    print(f"Dataset Summary:")
    print(f"Training pairs: {len(lr_train)}")
    print(f"Testing pairs: {len(lr_test)}")
    print(f"LR shape: {lr_shape}")
    print(f"HR shape: {hr_shape}")
    
    # Create models
    print("Building models...")
    
    # Input shape placeholders
    lr_ip = Input(shape=lr_shape)
    hr_ip = Input(shape=hr_shape)
    
    # Create generator
    generator = create_gen(lr_ip, num_res_block=16)
    generator.summary()
    
    # Create discriminator
    discriminator = create_disc(hr_ip)
    discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    discriminator.summary()
    
    # Create VGG19 model for perceptual loss
    vgg = build_vgg(hr_shape)
    vgg.trainable = False
    
    # Create combined GAN model
    gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
    gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")
    gan_model.summary()
    
    # Create batches for training
    train_lr_batches = []
    train_hr_batches = []
    for it in range(int(hr_train.shape[0] // batch_size)):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_hr_batches.append(hr_train[start_idx:end_idx])
        train_lr_batches.append(lr_train[start_idx:end_idx])
    
    # Initialize history tracking
    g_losses = []
    d_losses = []
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    
    for e in range(epochs):
        fake_label = np.zeros((batch_size, 1))  # Label for fake images
        real_label = np.ones((batch_size, 1))   # Label for real images
        
        epoch_g_losses = []
        epoch_d_losses = []
        
        # Train on batches
        for b in tqdm(range(len(train_hr_batches)), desc=f"Epoch {e+1}/{epochs}"):
            lr_imgs = train_lr_batches[b]  # Batch of LR images
            hr_imgs = train_hr_batches[b]  # Batch of HR images
            
            # Generate fake images
            fake_imgs = generator.predict_on_batch(lr_imgs)
            
            # Train discriminator
            discriminator.trainable = True
            d_loss_gen = discriminator.train_on_batch(fake_imgs, fake_label)
            d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
            
            # Average discriminator loss
            d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
            
            # Train generator
            discriminator.trainable = False
            
            # Extract VGG features for perceptual loss
            image_features = vgg.predict(hr_imgs)
            
            # Train the GAN model
            g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])
            
            # Track losses
            epoch_d_losses.append(d_loss[0])
            epoch_g_losses.append(g_loss)
            
            # Sample images at intervals
            if b % sample_interval == 0:
                # Sample and save generated image
                idx = random.randint(0, len(lr_test)-1)
                sample_lr = np.expand_dims(lr_test[idx], axis=0)
                sample_hr = np.expand_dims(hr_test[idx], axis=0)
                sample_gen = generator.predict(sample_lr)
                
                # Save the images
                plt.figure(figsize=(16, 6))
                
                plt.subplot(131)
                plt.title('Low Resolution')
                plt.imshow(sample_lr[0])
                
                plt.subplot(132)
                plt.title('Generated High Resolution')
                plt.imshow(sample_gen[0])
                
                plt.subplot(133)
                plt.title('Original High Resolution')
                plt.imshow(sample_hr[0])
                
                plt.savefig(f"results/sample_epoch_{e+1}_batch_{b}.png")
                plt.close()
        
        # Compute average epoch losses
        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_loss = np.mean(epoch_d_losses)
        
        # Track overall losses
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"Epoch {e+1}/{epochs} - Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}")
        
        # Save model at intervals
        if (e+1) % save_interval == 0:
            generator.save(f"models/generator_epoch_{e+1}.h5")
            discriminator.save(f"models/discriminator_epoch_{e+1}.h5")
            
            # Plot and save loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(g_losses, label='Generator Loss')
            plt.plot(d_losses, label='Discriminator Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('SRGAN Training Losses')
            plt.savefig(f"results/loss_plot_epoch_{e+1}.png")
            plt.close()
    
    # Save final models
    generator.save("models/generator_final.h5")
    discriminator.save("models/discriminator_final.h5")
    
    # Plot final training curves
    plt.figure(figsize=(10, 6))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('SRGAN Training Losses')
    plt.savefig("results/final_loss_plot.png")
    plt.close()
    
    print("Training completed successfully!")
    return generator, discriminator, lr_test, hr_test

# Evaluate trained model
def evaluate_model(generator_path="models/generator_final.h5", num_samples=5):
    """
    Evaluate the trained SRGAN model
    
    Parameters:
    generator_path - Path to the trained generator model
    num_samples - Number of samples to evaluate
    """
    print(f"Evaluating model: {generator_path}")
    
    # Check if model exists
    if not os.path.exists(generator_path):
        print(f"Model not found at {generator_path}")
        return
    
    # Load the generator model
    generator = load_model(generator_path, compile=False)
    print("Generator model loaded successfully.")
    
    # Load test data
    try:
        lr_list = sorted(os.listdir("data/lr_images"))
        hr_list = sorted(os.listdir("data/hr_images"))
        
        # Use only a subset for testing
        lr_list = lr_list[-min(num_samples, len(lr_list)):]
        hr_list = hr_list[-min(num_samples, len(hr_list)):]
        
        lr_test = []
        hr_test = []
        
        for img_lr, img_hr in zip(lr_list, hr_list):
            # Load images
            img_lr = cv2.imread(os.path.join("data/lr_images", img_lr))
            img_hr = cv2.imread(os.path.join("data/hr_images", img_hr))
            
            # Convert BGR to RGB
            img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
            img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
            
            lr_test.append(img_lr)
            hr_test.append(img_hr)
        
        # Convert to numpy arrays
        lr_test = np.array(lr_test) / 255.0
        hr_test = np.array(hr_test) / 255.0
        
    except FileNotFoundError:
        print("Test data not found. Generating test images...")
        # Generate test images
        lr_test = []
        hr_test = []
        
        for i in range(num_samples):
            lr_img, hr_img = generate_test_image(filename=f"test_{i}", size=128)
            lr_test.append(cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB))
            hr_test.append(cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB))
        
        # Convert to numpy arrays
        lr_test = np.array(lr_test) / 255.0
        hr_test = np.array(hr_test) / 255.0
    
    # Evaluate model on test images
    print(f"Generating super-resolution images for {len(lr_test)} test samples...")
    
    # Compute metrics
    psnr_values = []
    ssim_values = []
    
    plt.figure(figsize=(15, 5*len(lr_test)))
    
    for i in range(len(lr_test)):
        # Get a sample
        sample_lr = np.expand_dims(lr_test[i], axis=0)
        sample_hr = hr_test[i]
        
        # Generate super-resolution image
        start_time = cv2.getTickCount()
        sample_sr = generator.predict(sample_lr)[0]
        end_time = cv2.getTickCount()
        
        # Calculate processing time
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        
        # Convert to uint8 for metric calculation
        sample_sr_uint8 = (sample_sr * 255).astype(np.uint8)
        sample_hr_uint8 = (sample_hr * 255).astype(np.uint8)
        
        # Calculate PSNR (Peak Signal-to-Noise Ratio)
        psnr = cv2.PSNR(sample_sr_uint8, sample_hr_uint8)
        psnr_values.append(psnr)
        
        # Calculate SSIM (Structural Similarity Index)
        sample_sr_gray = cv2.cvtColor(sample_sr_uint8, cv2.COLOR_RGB2GRAY)
        sample_hr_gray = cv2.cvtColor(sample_hr_uint8, cv2.COLOR_RGB2GRAY)
        ssim = cv2.compareSSIM(sample_sr_gray, sample_hr_gray)
        ssim_values.append(ssim)
        
        # Display the results
        plt.subplot(len(lr_test), 3, i*3 + 1)
        plt.title(f"Low Resolution")
        plt.imshow(lr_test[i])
        plt.axis('off')
        
        plt.subplot(len(lr_test), 3, i*3 + 2)
        plt.title(f"Super Resolution\nPSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}")
        plt.imshow(sample_sr)
        plt.axis('off')
        
        plt.subplot(len(lr_test), 3, i*3 + 3)
        plt.title("Original High Resolution")
        plt.imshow(sample_hr)
        plt.axis('off')
        
        print(f"Sample {i+1} - PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}, Time: {processing_time:.3f}s")
    
    plt.tight_layout()
    plt.savefig("results/evaluation_results.png")
    plt.close()
    
    # Print average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    print("\nEvaluation Results:")
    print(f"Average PSNR: {avg_psnr:.2f}dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Create a summary plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(121)
    plt.bar(range(len(psnr_values)), psnr_values)
    plt.axhline(y=avg_psnr, color='r', linestyle='--', label=f'Avg: {avg_psnr:.2f}dB')
    plt.xlabel('Sample')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Values')
    plt.legend()
    
    plt.subplot(122)
    plt.bar(range(len(ssim_values)), ssim_values)
    plt.axhline(y=avg_ssim, color='r', linestyle='--', label=f'Avg: {avg_ssim:.4f}')
    plt.xlabel('Sample')
    plt.ylabel('SSIM')
    plt.title('SSIM Values')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("results/evaluation_metrics.png")
    plt.close()
    
    print("Evaluation completed successfully!")

# Main function to control the entire process
def main():
    # Create directory structure
    setup_directories()
    
    # Generate dataset
    # You can provide your own source directory with images
    # or leave it empty to generate random images
    source_dir = "source_images"  # Change this to your image directory
    generate_dataset(source_dir, n_images=100, hr_size=128, lr_size=32)
    
    # Generate a test image
    generate_test_image()
    
    # Train the model
    # Adjust epochs for better results (recommended: 100+ for production)
    generator, discriminator, lr_test, hr_test = train_srgan(epochs=5, batch_size=1, 
                                                            sample_interval=50, 
                                                            save_interval=1)
    
    # Evaluate the model
    evaluate_model(generator_path="models/generator_final.h5", num_samples=5)
    
    print("SRGAN process")
