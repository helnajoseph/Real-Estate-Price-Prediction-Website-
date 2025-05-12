$(document).ready(function() {
    $('#prediction-form').on('submit', function(e) {
        e.preventDefault();

        // Show loading state
        $('#prediction-form button[type="submit"]').html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...');
        $('#prediction-form button[type="submit"]').prop('disabled', true);

        // Get form data
        const formData = $(this).serialize();

        // Send AJAX request
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: formData,
            success: function(response) {
                if (response.success) {
                    // Format the predictions with commas for thousands
                    const formattedRupees = new Intl.NumberFormat('en-IN', {
                        maximumFractionDigits: 0
                    }).format(response.prediction_rupees);

                    const formattedLakhs = new Intl.NumberFormat('en-IN', {
                        maximumFractionDigits: 2
                    }).format(response.prediction_lakhs);

                    // Show the results
                    $('#prediction-rupees').text(formattedRupees);
                    $('#prediction-lakhs').text(formattedLakhs);
                    $('#result').fadeIn();

                    // Scroll to result
                    $('html, body').animate({
                        scrollTop: $('#result').offset().top - 100
                    }, 500);
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function(xhr) {
                let errorMessage = 'An error occurred during prediction.';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                }
                alert('Error: ' + errorMessage);
            },
            complete: function() {
                // Reset button state
                $('#prediction-form button[type="submit"]').html('Predict Price');
                $('#prediction-form button[type="submit"]').prop('disabled', false);
            }
        });
    });
});
