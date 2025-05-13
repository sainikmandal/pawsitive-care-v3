import { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  FormControl,
  FormHelperText,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
  Alert,
  Chip,
  Stack,
} from '@mui/material';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { predictDisease } from '../services/api';
import type { PredictionResponse, Animal, Symptom } from '../types';
import { ANIMALS, SYMPTOMS } from '../types';

const predictionSchema = z.object({
  animal: z.string().min(1, 'Please select an animal'),
  age: z.number().min(0, 'Age must be positive').max(30, 'Age seems too high'),
  temperature: z.number().min(95, 'Temperature too low').max(110, 'Temperature too high'),
  symptoms: z.array(z.string()).min(1, 'Select at least one symptom').max(3, 'Maximum 3 symptoms allowed'),
});

type PredictionFormData = z.infer<typeof predictionSchema>;

export const PredictionForm = () => {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm<PredictionFormData>({
    resolver: zodResolver(predictionSchema),
    defaultValues: {
      animal: '',
      age: 0,
      temperature: 98.6,
      symptoms: [],
    },
  });

  const onSubmit = async (data: PredictionFormData) => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictDisease(data);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Card>
        <CardContent>
          <Typography variant="h4" component="h1" gutterBottom>
            Livestock Disease Prediction
          </Typography>
          
          <form onSubmit={handleSubmit(onSubmit)}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                <Box sx={{ flex: '1 1 300px', minWidth: 0 }}>
                  <Controller
                    name="animal"
                    control={control}
                    render={({ field }) => (
                      <FormControl fullWidth error={!!errors.animal}>
                        <InputLabel>Animal Type</InputLabel>
                        <Select {...field} label="Animal Type">
                          {ANIMALS.map((animal: Animal) => (
                            <MenuItem key={animal.value} value={animal.value}>
                              {animal.label}
                            </MenuItem>
                          ))}
                        </Select>
                        {errors.animal && (
                          <FormHelperText>{errors.animal.message}</FormHelperText>
                        )}
                      </FormControl>
                    )}
                  />
                </Box>

                <Box sx={{ flex: '1 1 300px', minWidth: 0 }}>
                  <Controller
                    name="age"
                    control={control}
                    render={({ field }) => (
                      <TextField
                        {...field}
                        type="number"
                        label="Age (years)"
                        fullWidth
                        error={!!errors.age}
                        helperText={errors.age?.message}
                        onChange={(e) => field.onChange(Number(e.target.value))}
                      />
                    )}
                  />
                </Box>
              </Box>

              <Box sx={{ flex: '1 1 300px', minWidth: 0 }}>
                <Controller
                  name="temperature"
                  control={control}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      type="number"
                      label="Temperature (°F)"
                      fullWidth
                      error={!!errors.temperature}
                      helperText={errors.temperature?.message}
                      onChange={(e) => field.onChange(Number(e.target.value))}
                    />
                  )}
                />
              </Box>

              <Box sx={{ flex: '1 1 100%', minWidth: 0 }}>
                <Controller
                  name="symptoms"
                  control={control}
                  render={({ field }) => (
                    <FormControl fullWidth error={!!errors.symptoms}>
                      <InputLabel>Symptoms</InputLabel>
                      <Select
                        {...field}
                        multiple
                        label="Symptoms"
                        renderValue={(selected) => (
                          <Stack direction="row" spacing={1} flexWrap="wrap">
                            {selected.map((value) => (
                              <Chip
                                key={value}
                                label={SYMPTOMS.find((s) => s.value === value)?.label}
                                size="small"
                              />
                            ))}
                          </Stack>
                        )}
                      >
                        {SYMPTOMS.map((symptom: Symptom) => (
                          <MenuItem key={symptom.value} value={symptom.value}>
                            {symptom.label}
                          </MenuItem>
                        ))}
                      </Select>
                      <FormHelperText>
                        Select up to 3 symptoms
                        {errors.symptoms && ` - ${errors.symptoms.message}`}
                      </FormHelperText>
                    </FormControl>
                  )}
                />
              </Box>

              <Box>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  fullWidth
                  disabled={loading}
                  sx={{ mt: 2 }}
                >
                  {loading ? <CircularProgress size={24} /> : 'Predict Disease'}
                </Button>
              </Box>
            </Box>
          </form>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          {prediction && (
            <Box sx={{ mt: 4 }}>
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    Prediction Results
                  </Typography>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="h6" color="primary">
                      {prediction.predicted_disease}
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  {/* Warning Section */}
                  <Box sx={{ mt: 3, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
                    <Typography variant="h6" color="warning.dark" gutterBottom>
                      ⚠️ Important Notice
                    </Typography>
                    <Typography variant="body1" color="warning.dark" paragraph>
                      This prediction is based on machine learning and should not be considered a definitive diagnosis. 
                      Please consult with a qualified veterinarian immediately for proper medical attention.
                    </Typography>
                    <Typography variant="body2" color="warning.dark" paragraph>
                      Early intervention can significantly improve treatment outcomes. Don't delay seeking professional help.
                    </Typography>
                    <Button
                      variant="contained"
                      color="warning"
                      href="/vets"
                      sx={{ mt: 1 }}
                    >
                      Find a Trusted Veterinarian
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}; 