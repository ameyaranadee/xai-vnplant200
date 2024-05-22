$(document).ready(function() {
    // Load the initial segmentation results
    loadSegmentationResults(1.0, 10);

    // Event listener for the ratio input
    $('#ratio-input').on('input', function() {
      const ratio = $(this).val();
      const max_dist = $('#max-dist-input').val();
      loadSegmentationResults(ratio, max_dist);
    });
    // Event listener for the max_dist input
    $('#max-dist-input').on('input', function() {
      const ratio = $('#ratio-input').val();
      const max_dist = $(this).val();
      loadSegmentationResults(ratio, max_dist);
    });
  });

  function loadSegmentationResults(ratio, max_dist) {
    $.ajax({
      url: '/segmentation',
      data: { ratio: ratio, max_dist: max_dist },
      success: function(data) {
        $('#segmented-image-ratio').attr('src', `data:image/png;base64,${data[0]}`);
        $('#segmented-image-max-dist').attr('src', `data:image/png;base64,${data[1]}`);
      }
    });
  }

//   function displayUploadedImage(event) {
//     const imageInput = event.target;
//     const imageFile = imageInput.files[0];
  
//     if (!imageFile) {
//       alert('Please select an image file.');
//       return;
//     }
  
//     const reader = new FileReader();
//     reader.onload = function() {
//       $('#uploaded-image').attr('src', reader.result);
//       $('#uploaded-image').show();
//       $('#predict-button').prop('disabled', false);
//     }
//     reader.readAsDataURL(imageFile);
//   }
  
//   $('#predict-button').click(function() {
//     const imageInput = $('#image-input')[0];
//     const imageFile = imageInput.files[0];
  
//     if (!imageFile) {
//       alert('Please select an image file.');
//       return;
//     }
  
//     const formData = new FormData();
//     formData.append('image', imageFile);
  
//     $.ajax({
//       url: '/predict',
//       type: 'POST',
//       data: formData,
//       processData: false,
//       contentType: false,
//       success: function(data) {
//         const predictedClass = data.predicted_class;
//         $('#prediction-result').text(`Predicted Class: ${predictedClass}`);
//       },
//       error: function() {
//         alert('An error occurred during prediction.');
//       }
//     });
//   });

function generateLimeExplanation(event) {
    const imageInput = event.target;
    const imageFile = imageInput.files[0];
  
    if (!imageFile) {
      alert('Please select an image file.');
      return;
    }
  
    const reader = new FileReader();
    reader.onload = function() {
      $('#uploaded-image').attr('src', reader.result);
      $('#uploaded-image').show();
  
      const formData = new FormData();
      formData.append('image', imageFile);
      formData.append('positive_only', $('#positive-only-input').prop('checked'));
      formData.append('num_features', $('#num-features-input').val());
      formData.append('hide_rest', $('#hide-rest-input').prop('checked'));
  
      $.ajax({
        url: '/explain',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(data) {
          const predictedClass = data.predicted_class;
          const explanationImage = data.explanation_image;
          $('#explanation-result').text(`Predicted Class: ${predictedClass}`);
          $('#explanation-image').attr('src', `data:image/png;base64,${explanationImage}`);
        },
        error: function() {
          alert('An error occurred during LIME explanation.');
        }
      });
    }
    reader.readAsDataURL(imageFile);
  }
