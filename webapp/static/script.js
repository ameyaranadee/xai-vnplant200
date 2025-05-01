// $(document).ready(function() {
//     // Load the initial segmentation results
//     loadSegmentationResults(1.0, 10);

//     // Update slider values on page load
//     $('#ratio-value').text($('#ratio-input').val());
//     $('#max-dist-value').text($('#max-dist-input').val());
//     $('#num-features-value').text($('#num-features-input').val());

//     // Event listener for the ratio input
//     $('#ratio-input').on('input', function() {
//       const ratio = $(this).val();
//       $('#ratio-value').text(ratio);
//       const max_dist = $('#max-dist-input').val();
//       loadSegmentationResults(ratio, max_dist);
//     });
//     // Event listener for the max_dist input
//     $('#max-dist-input').on('input', function() {
//       const ratio = $('#ratio-input').val();
//       const max_dist = $(this).val();
//       $('#max-dist-value').text(max_dist);
//       loadSegmentationResults(ratio, max_dist);
//     });
//   });

//   function loadSegmentationResults(ratio, max_dist) {
//     $.ajax({
//       url: '/segmentation',
//       data: { ratio: ratio, max_dist: max_dist },
//       success: function(data) {
//         $('#original-image').attr('src', `data:image/png;base64,${data[0]}`);
//         $('#segmented-image-ratio').attr('src', `data:image/png;base64,${data[1]}`);
//         $('#segmented-image-max-dist').attr('src', `data:image/png;base64,${data[2]}`);
//       }
//     });
//   }


// function generateLimeExplanation(event) {
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
  
//       const formData = new FormData();
//       formData.append('image', imageFile);
//       formData.append('positive_only', $('#positive-only-input').prop('checked'));
//       formData.append('num_features', $('#num-features-input').val());
//       formData.append('hide_rest', $('#hide-rest-input').prop('checked'));
  
//       $.ajax({
//         url: '/explain',
//         type: 'POST',
//         data: formData,
//         processData: false,
//         contentType: false,
//         beforeSend: function() {
//           // Show loader or loading message
//           // For simplicity, let's assume a loader with a CSS class
//           $('#explanation-image').hide();
//           $('#explanation-result').hide();
//           $('#loader').show(); // Assuming you have a loader element
//         },
        // success: function(data) {
        //   // const predictedClass = data.predicted_class;
        //   const predictedClass = data.top_classes ? data.top_classes[0] : 'undefined';
        //   // const explanationImage = data.explanation_image;
        //   const explanationImage = data.explanations ? data.explanations[0] : ' '; 
        //   $('#explanation-result').text(`Predicted Class: ${predictedClass}`).show();
        //   if (explanationImage) {
        //     $('#explanation-image').attr('src', `data:image/png;base64,${explanationImage}`).show();
        //     // $('#explanation-image')
        //   } else {
        //     $('#explanation-image').hide();
        //     alert('No explanation image available.');
        //   }
          
        //   // $('#explanation-result');
        //   $('#loader').hide();
//         },
//         error: function() {
//           alert('An error occurred during LIME explanation.');
//           $('#loader').hide();
//         }
//       });
//     }
//     reader.readAsDataURL(imageFile);
//   }

$(document).ready(function () {
  loadSegmentationResults(1.0, 10);

  $('#ratio-input, #max-dist-input').on('input', function () {
    const ratio = $('#ratio-input').val();
    const maxDist = $('#max-dist-input').val();
    loadSegmentationResults(ratio, maxDist);
  });

  $('#model-select').on('change', function () {
    if ($(this).val() === 'inception') {
      $('#lime-params').show();
    } else {
      $('#lime-params').hide();
    }
  }).trigger('change');
});

function loadSegmentationResults(ratio, max_dist) {
  $.ajax({
    url: '/segmentation',
    data: { ratio, max_dist },
    success: function (data) {
      $('#original-image').attr('src', `data:image/png;base64,${data[0]}`);
      $('#segmented-image-ratio').attr('src', `data:image/png;base64,${data[1]}`);
      $('#segmented-image-max-dist').attr('src', `data:image/png;base64,${data[2]}`);
    }
  });
}

function generateExplanation(event) {
    const file = event.target.files[0];
    const model = document.getElementById('model-select').value;
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model', model);
    if (model == 'inception') {
        formData.append('num_features', $('#num-features-input').val() || 5);
        formData.append('positive_only', $('#positive-only-input').prop('checked'));
        formData.append('hide_rest', $('#hide-rest-input').prop('checked'));
    }

    const reader = new FileReader();
    reader.onload = function(e) {
        $('#uploaded-image').attr('src', e.target.result).show();
    }
    reader.readAsDataURL(file);
    fetch('/explain', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
      const predictedClass = data.top_classes ? data.top_classes[0] : 'undefined';
      const explanationImage = data.explanations ? data.explanations[0] : '';

      $('#explanation-result').text(`Predicted Class: ${predictedClass}`).show();

      if (explanationImage) {
        $('#explanation-image').attr('src', `data:image/png;base64,${explanationImage}`).show();
      } else {
        $('#explanation-image').hide();
        alert('No explanation image available.');
      }
    })
    .catch(error => {
      console.error('Error generating explanation:', error);
      alert('An error occurred while generating the explanation.');
    });
}

// Function to adjust params and update the explanation image
function adjustParameters(numFeatures, positiveOnly, hideRest) {
  $.ajax({
      url: '/adjust_parameters',
      type: 'POST',
      data: {
          num_features: numFeatures,
          positive_only: positiveOnly,
          hide_rest: hideRest
      },
      beforeSend: function() {
        // Show loader or loading message
        // For simplicity, let's assume a loader with a CSS class
        $('#explanation-image').hide();
        $('#loader').show(); // Assuming you have a loader element
      },
      success: function(data) {
          $('#explanation-image').attr('src', `data:image/png;base64,${data.new_explanation_image}`);
          $('#explanation-image').show();
          $('#loader').hide();
      },
      error: function() {
          alert('An error occurred while adjusting parameters.');
          $('#loader').hide();
      }
  });
}



// // Event listener for adjusting parameters based on user input
// $('#num-features-input').on('input', function() {
//   const numFeatures = $(this).val();
//   $('#num-features-value').text(numFeatures);
//   adjustParameters(numFeatures, $('#positive-only-input').prop('checked'), $('#hide-rest-input').prop('checked'));
// });

// $('#positive-only-input, #hide-rest-input').on('change', function() {
//   const numFeatures = $('#num-features-input').val();
//   adjustParameters(numFeatures, $('#positive-only-input').prop('checked'), $('#hide-rest-input').prop('checked'));
// });
