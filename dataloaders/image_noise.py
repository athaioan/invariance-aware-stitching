import torch
from torchvision.transforms.functional import to_pil_image

class ImageNoise():

    def __init__(self, num_classes, noise_type, height, width, num_channels=3, dataset_size=10_000, transform=None, joint_transform=None, target_transform=None, use_multi_crop=False, K_init=None):

        self.noise_type = noise_type
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.dataset_size = dataset_size
        self.num_classes = num_classes

        self.use_multi_crop = use_multi_crop
        self.transform = transform
        self.joint_transform = joint_transform
        self.target_transform = target_transform
        self.targets = None

        if K_init is not None:
            self.dataset_size = K_init


    def __getitem__(self, _):

        if self.noise_type == 'freq':
            img = self.generate_1_over_f_noise_rgb()
        elif self.noise_type == 'uniform':
            img = self.uniform_noise()
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")


        target = torch.randint(0, self.num_classes, (1,)).item()

        if self.joint_transform is not None and self.joint_transform.backdoor_first:
            img = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)

        if self.joint_transform is not None and not(self.joint_transform.backdoor_first):
            img = self.joint_transform(img, target)

        if self.target_transform is not None:
            target = self.target_transform(target)


        return img, target


    def __len__(self):
       
       return self.dataset_size
    
    def __repr__(self):

        return f"Dataset {self.noise_type} Noise\n \
                 Height: {self.height}, Width: {self.width}, \n \
                 Number of datapoints: {self.dataset_size} \n \
                 StandardTransform: {self.transform if self.transform else 'None'} \n"
            
    def uniform_noise(self):
        # Generate uniform noise
        img = torch.rand(self.num_channels, self.height, self.width)

        img = to_pil_image(img)
        return img

    def generate_1_over_f_noise_rgb(self, device='cpu'):

        def _generate_single_channel(height, width):
            # Frequency grid
            fy = torch.fft.fftfreq(height, device=device).reshape(-1, 1)
            fx = torch.fft.fftfreq(width, device=device).reshape(1, -1)

            freqs = torch.sqrt(fx**2 + fy**2)
            freqs[0, 0] = 1.0  # avoid division by zero

            spectrum = 1.0 / freqs

            # Random phase
            phase = torch.rand(height, width, device=device) * 2 * torch.pi
            phase = torch.polar(torch.ones_like(phase), phase)

            fourier = spectrum * phase
            noise = torch.fft.ifft2(fourier).real
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            return noise

        
        r = _generate_single_channel(self.height, self.width)
        g = _generate_single_channel(self.height, self.width)
        b = _generate_single_channel(self.height, self.width)

        if self.num_channels == 1:
            img = torch.stack([r], dim=0)  # Shape: [1, H, W]
        elif self.num_channels == 3:
            img = torch.stack([r, g, b], dim=0)  # Shape: [3, H, W]

        # Convert to PIL Image
        img = to_pil_image(img)
        
        return img