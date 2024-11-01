// static/js/script.js
const config = {
    maxImageSize: 10 * 1024 * 1024, // 10MB
    apiUrl: window.location.origin,
    timeouts: {
        ocr: 180000     // 3 minutes
    },
    imageMaxDimension: 2000,
    imageQuality: 0.8
};

// Elements
const elements = {
    imageInput: document.getElementById('imageInput'),
    preview: document.getElementById('preview'),
    ocrButton: document.getElementById('ocrButton'),
    ocrImage: document.getElementById('ocrImage'),
    ocrText: document.getElementById('ocrText'),
    textResults: document.getElementById('textResults'),
    loading: document.getElementById('loading')
};

// Utility functions
const utils = {
    toggleLoading: (show) => {
        elements.loading.style.display = show ? 'block' : 'none';
        document.body.style.cursor = show ? 'wait' : 'default';
    },

    handleError: (error) => {
        console.error('Error:', error);
        const message = error.response?.data?.detail 
            || error.response?.statusText 
            || error.message 
            || 'An unexpected error occurred';
        alert(message);
        utils.toggleLoading(false);
    },

    resizeImage: async (file) => {
        const image = await createImageBitmap(file);
        const canvas = document.createElement('canvas');
        let { width, height } = image;

        if (width > config.imageMaxDimension || height > config.imageMaxDimension) {
            const aspectRatio = width / height;
            if (width > height) {
                width = config.imageMaxDimension;
                height = Math.round(config.imageMaxDimension / aspectRatio);
            } else {
                height = config.imageMaxDimension;
                width = Math.round(config.imageMaxDimension * aspectRatio);
            }
        }

        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, width, height);
        return canvas.toDataURL('image/jpeg', config.imageQuality);
    },

    displayResults: (imageData, textData) => {
        // Display processed image
        elements.ocrImage.src = `data:image/jpeg;base64,${imageData}`;
        elements.ocrImage.style.display = 'block';

        // Display text results
        elements.textResults.innerHTML = textData
            .map(text => `<div class="text-line">${text}</div>`)
            .join('');
        elements.ocrText.style.display = 'block';
        
        // Scroll to results
        elements.ocrImage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
};

// Event handlers
const handlers = {
    imageUpload: async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        try {
            if (file.size > config.maxImageSize) {
                throw new Error(`Image size must be less than ${config.maxImageSize / (1024 * 1024)}MB`);
            }

            utils.toggleLoading(true);
            const resizedImage = await utils.resizeImage(file);
            elements.preview.src = resizedImage;
            elements.preview.style.display = 'block';
            elements.ocrButton.disabled = false;
            
            // Clear previous results
            elements.ocrImage.style.display = 'none';
            elements.ocrText.style.display = 'none';
            
        } catch (error) {
            utils.handleError(error);
        } finally {
            utils.toggleLoading(false);
        }
    },

    processOCR: async () => {
        try {
            elements.ocrButton.disabled = true;
            utils.toggleLoading(true);

            const imageData = elements.preview.src.split(',')[1];
            const response = await axios.post(`${config.apiUrl}/ocr`, 
                { img: imageData },
                {
                    headers: { 'Content-Type': 'application/json' },
                    timeout: config.timeouts.ocr
                }
            );

            if (response.data?.img && response.data?.text) {
                utils.displayResults(response.data.img, response.data.text);
            } else {
                throw new Error('Invalid response format from server');
            }

        } catch (error) {
            utils.handleError(error);
        } finally {
            elements.ocrButton.disabled = false;
            utils.toggleLoading(false);
        }
    }
};

// Event listeners
elements.imageInput.addEventListener('change', handlers.imageUpload);
elements.ocrButton.addEventListener('click', handlers.processOCR);