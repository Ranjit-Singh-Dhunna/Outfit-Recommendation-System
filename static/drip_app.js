document.addEventListener('DOMContentLoaded', function() {
    // File upload elements
    const fileInput = document.getElementById('file-input');
    const fileStatus = document.getElementById('file-status');
    const previewContainer = document.getElementById('preview-container');
    const folderContainer = document.querySelector('.folder-container');
    
    // Form elements
    const generateBtn = document.getElementById('generate-btn');
    const suggestionsDiv = document.getElementById('suggestions');
    
    // Data storage
    let uploadedImages = [];
    let metadata = [];
    let currentOutfits = [];

    // ======================
    // FILE UPLOAD HANDLING
    // ======================

    function updateFileStatus(count) {
        if (count === 0) {
            fileStatus.textContent = 'No files selected';
            folderContainer.style.background = 'linear-gradient(135deg, #6dd5ed, #2193b0)';
        } else {
            fileStatus.textContent = `${count} file${count > 1 ? 's' : ''} selected`;
            folderContainer.style.background = 'linear-gradient(135deg, #4CAF50, #2E8B57)';
        }
    }

    function createImagePreview(file, fileUrl) {
        const previewItem = document.createElement('div');
        previewItem.className = 'preview-item';
        
        const img = document.createElement('img');
        img.src = fileUrl;
        img.className = 'preview-image';
        img.title = file.name;
        
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-btn';
        removeBtn.innerHTML = '×';
        removeBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            previewItem.remove();
            uploadedImages = uploadedImages.filter(f => f !== file);
            updateFileStatus(uploadedImages.length);
        });
        
        previewItem.appendChild(img);
        previewItem.appendChild(removeBtn);
        previewContainer.appendChild(previewItem);
    }

    fileInput.addEventListener('change', function(e) {
        const files = Array.from(e.target.files);
        
        if (files.length === 0) return;
        
        previewContainer.innerHTML = '';
        uploadedImages = [];
        
        files.forEach(file => {
            if (!file.type.match('image.*')) {
                console.warn(`Skipped non-image file: ${file.name}`);
                return;
            }
            
            const reader = new FileReader();
            reader.onload = function(e) {
                createImagePreview(file, e.target.result);
                uploadedImages.push(file);
                updateFileStatus(uploadedImages.length);
            };
            reader.readAsDataURL(file);
        });
    });

    folderContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        folderContainer.style.background = 'linear-gradient(135deg, #2196F3, #0d8aee)';
    });
    
    folderContainer.addEventListener('dragleave', () => {
        updateFileStatus(uploadedImages.length);
    });
    
    folderContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });

   folderContainer.addEventListener('click', function(e) {
    // Don't trigger if the click is on the label or input
    if (e.target === fileInput || e.target.closest('.custom-file-upload')) return;
    fileInput.click();
    e.stopPropagation();
});

    function validateForm() {
        if (uploadedImages.length === 0) {
            alert('Please upload at least one image first');
            return false;
        }
        
        const gender = document.getElementById('gender').value.trim();
        const skinTone = document.getElementById('skin-tone').value.trim();
        const bodyType = document.getElementById('body-type').value.trim();
        const style = document.getElementById('style').value.trim();
        const occasion = document.getElementById('occasion').value.trim();
        const facial = document.getElementById('facial').value.trim();
        const num = document.getElementById('num').value.trim();
        const message = document.getElementById('message').value.trim();
        
        const numValue = Number(num);
        if (isNaN(numValue)) {
            alert('Please enter a valid number');
            return false;
        }
        
        if (numValue < 1) {
            alert('Please enter a number greater than 0');
            return false;
        }
        
        if (!gender || !skinTone || !bodyType || !style || !occasion) {
            alert('Please complete all required profile fields');
            return false;
        }
        
        return {
            gender,
            skinTone,
            bodyType,
            style,
            occasion,
            facial,
            num: numValue,
            message
        };
    }

    // ======================
    // API COMMUNICATION
    // ======================

    async function uploadFiles() {
        const formData = new FormData();
        uploadedImages.forEach(file => {
            formData.append('files', file);
        });
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `Upload failed (${response.status})`);
        }
        
        return await response.json();
    }

    async function getSuggestions(metadata, userProfile) {
        try {
            const response = await fetch('/api/suggestions', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    metadata,
                    userProfile
                })
            });
            
            const data = await response.json();
            
            if (!response.ok || !data.success) {
                throw new Error(data.error || `Suggestion failed (${response.status})`);
            }
            
            return data;
        } catch (error) {
            console.error('Error getting suggestions:', error);
            throw error;
        }
    }

    // ======================
    // UI UPDATES
    // ======================

    function showLoading() {
        suggestionsDiv.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Analyzing your wardrobe and preferences...</p>
                <p class="loading-subtext">Creating perfect outfit combinations</p>
            </div>
        `;
    }

    function displaySuggestions(responseData) {
    if (!responseData.outfits || responseData.outfits.length === 0) {
        throw new Error("No outfit suggestions were generated");
    }

    // Store current outfits
    currentOutfits = responseData.outfits;
    
    suggestionsDiv.innerHTML = responseData.outfits.map((outfit, index) => `
        <div class="outfit">
            <h3>${outfit.outfit_name}</h3>
            <div class="outfit-items-preview">
                ${outfit.items.map(item => `
                    <div class="outfit-item-preview">
                        <img src="/uploads/${encodeURIComponent(item)}" alt="Clothing item">
                        <p>${item.split('/').pop().replace(/\.[^/.]+$/, "")}</p>
                    </div>
                `).join('')}
            </div>
            <div class="outfit-reasoning">
                <p>${outfit.reasoning}</p>
            </div>
        </div>
    `).join('<hr>');

    if (!responseData.aiUsed) {
        suggestionsDiv.innerHTML += `
            <div class="notice">
                <p>Basic suggestions - AI service unavailable</p>
            </div>
        `;
    }
}
    function showError(message) {
        suggestionsDiv.innerHTML = `
            <div class="error">
                <svg viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
                </svg>
                <h3>Oops! Something went wrong</h3>
                <p>${message}</p>
                <button id="retry-btn" class="retry-btn">Try Again</button>
            </div>
        `;
        
        document.getElementById('retry-btn').addEventListener('click', generateSuggestions);
    }

    // ======================
    // MAIN FUNCTION
    // ======================

    async function generateSuggestions() {
        const valid = validateForm();
        if (!valid) return;
        
        showLoading();
        
        try {
            const uploadResponse = await uploadFiles();
            const suggestionsResponse = await getSuggestions(uploadResponse.metadata, valid);
            displaySuggestions(suggestionsResponse);
        } catch (error) {
            showError(error.message);
            console.error('Error generating suggestions:', error);
        }
    }

    // Event listeners
    generateBtn.addEventListener('click', generateSuggestions);
});