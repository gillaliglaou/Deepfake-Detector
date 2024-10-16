(function($) {

  "use strict";

  // init Chocolat light box
  var initChocolat = function() {
  Chocolat(document.querySelectorAll('.image-link'), {
      imageSize: 'contain',
      loop: true,
    })
  }

  // document ready
  $(document).ready(function(){
    initChocolat();
  });

})(jQuery);

// Function to handle image preview
function previewFile() {
  const fileInput = document.getElementById('fileInput');
  const imageContainer = document.getElementById('imageContainer');

  fileInput.addEventListener('change', function() {
    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function(event) {
      const img = document.createElement('img');
      img.src = event.target.result;
      img.alt = 'Preview';
      img.classList.add('banner-image');
      img.style.maxWidth = '100%';
      img.style.maxHeight = '100%';
      img.style.objectFit = 'contain';
      imageContainer.innerHTML = '';
      imageContainer.appendChild(img);
    };

    reader.readAsDataURL(file);
  });
}

// Call the function when the page is loaded
window.onload = function() {
  // Attach the previewFile function to the change event of the file input
  document.getElementById('fileInput').addEventListener('change', previewFile);
};
