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
  { value: 'depression', label: 'Depression' },
  { value: 'loss of appetite', label: 'Loss of Appetite' },
  { value: 'weight loss', label: 'Weight Loss' },
  { value: 'swelling', label: 'Swelling' },
  { value: 'swelling in limb', label: 'Swelling in Limb' },
  { value: 'crackling sound', label: 'Crackling Sound' },
  { value: 'lameness', label: 'Lameness' },
  { value: 'fever', label: 'Fever' },
  { value: 'coughing', label: 'Coughing' },
  { value: 'diarrhea', label: 'Diarrhea' },
  { value: 'vomiting', label: 'Vomiting' },
  { value: 'dehydration', label: 'Dehydration' },
  { value: 'lethargy', label: 'Lethargy' },
  { value: 'abnormal breathing', label: 'Abnormal Breathing' },
  { value: 'nasal discharge', label: 'Nasal Discharge' },
  { value: 'eye discharge', label: 'Eye Discharge' },
  { value: 'skin lesions', label: 'Skin Lesions' },
  { value: 'hair loss', label: 'Hair Loss' },
  { value: 'itching', label: 'Itching' },
  { value: 'abnormal gait', label: 'Abnormal Gait' },
  { value: 'stiffness', label: 'Stiffness' },
  { value: 'tremors', label: 'Tremors' },
  { value: 'seizures', label: 'Seizures' },
  { value: 'sudden death', label: 'Sudden Death' },
  { value: 'bloody discharge', label: 'Bloody Discharge' }
]; 