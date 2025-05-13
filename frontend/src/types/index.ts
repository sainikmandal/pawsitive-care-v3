export interface PredictionInput {
  animal: string;
  age: number;
  temperature: number;
  symptoms: string[];
}

export interface PredictionResponse {
  predicted_disease: string;
  confidence: number;
}

export interface HealthResponse {
  status: string;
  model_status: string;
}

export interface Animal {
  value: string;
  label: string;
}

export interface Symptom {
  value: string;
  label: string;
}

export const ANIMALS: Animal[] = [
  { value: 'cow', label: 'Cow' },
  { value: 'buffalo', label: 'Buffalo' },
  { value: 'sheep', label: 'Sheep' },
  { value: 'goat', label: 'Goat' },
];

export const SYMPTOMS: Symptom[] = [
  { value: 'loss of appetite', label: 'Loss of Appetite' },
  { value: 'depression', label: 'Depression' },
  { value: 'painless lumps', label: 'Painless Lumps' },
  { value: 'difficulty walking', label: 'Difficulty Walking' },
  { value: 'lameness', label: 'Lameness' },
  { value: 'crackling sound', label: 'Crackling Sound' },
  { value: 'fatigue', label: 'Fatigue' },
  { value: 'shortness of breath', label: 'Shortness of Breath' },
  { value: 'sweats', label: 'Sweats' },
  { value: 'chest discomfort', label: 'Chest Discomfort' },
  { value: 'chills', label: 'Chills' },
  { value: 'swelling in muscle', label: 'Swelling in Muscle' },
  { value: 'swelling in limb', label: 'Swelling in Limb' },
  { value: 'swelling in abdomen', label: 'Swelling in Abdomen' },
  { value: 'swelling in extremities', label: 'Swelling in Extremities' },
  { value: 'swelling in neck', label: 'Swelling in Neck' },
  { value: 'sores on tongue', label: 'Sores on Tongue' },
  { value: 'sores on gums', label: 'Sores on Gums' },
  { value: 'blisters on hooves', label: 'Blisters on Hooves' },
  { value: 'sores on mouth', label: 'Sores on Mouth' },
  { value: 'sores on hooves', label: 'Sores on Hooves' },
  { value: 'blisters on tongue', label: 'Blisters on Tongue' },
  { value: 'blisters on gums', label: 'Blisters on Gums' },
  { value: 'blisters on mouth', label: 'Blisters on Mouth' }
]; 