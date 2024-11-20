from django.db import models

class ArchitecturePrediction(models.Model):
    predicted_label = models.CharField(max_length=100)
    confidence_scores = models.TextField()  # Stores confidence scores as a text field
    image = models.ImageField(upload_to='uploads/')
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_label} - {self.timestamp}"
