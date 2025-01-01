# Use Miniconda as the base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Install dependencies for headless rendering (EGL and related libraries)
RUN apt-get update && apt-get install -y \
    libegl1-mesa \
    libgles2-mesa \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the environment file
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the environment by default and ensure PATH includes the environment
RUN echo "conda activate my_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/my_env/bin:$PATH

# Copy the rest of the application
COPY . .

# Make the training script executable
RUN chmod +x ./scripts/train.sh

# Run the training script to verify installation (optional; can be commented out after testing)
# RUN ./scripts/train.sh

# Expose a port if needed (adjust based on the application)
EXPOSE 8000

# Define the default command (adjust based on the application)
CMD ["./scripts/train.sh"]

