import axios, { AxiosError } from 'axios';
import type { PredictionInput, PredictionResponse, HealthResponse } from '../types';

// Use the Vite proxy URL
const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  timeout: 10000, // 10 second timeout
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log('Making request to:', config.url);
    console.log('Request data:', config.data);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log('Response received:', response.status);
    console.log('Response data:', response.data);
    return response;
  },
  (error) => {
    console.error('Response error:', error);
    if (error.response) {
      console.error('Error response data:', error.response.data);
      console.error('Error response status:', error.response.status);
    }
    return Promise.reject(error);
  }
);

export const predictDisease = async (input: PredictionInput): Promise<PredictionResponse> => {
  try {
    const response = await api.post<PredictionResponse>('/predict', input);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      if (axiosError.response) {
        const errorData = axiosError.response.data as { detail?: string };
        throw new Error(errorData.detail || 'Server error occurred');
      } else if (axiosError.request) {
        throw new Error('No response received from server. Please check if the backend is running.');
      } else {
        throw new Error('Request setup failed: ' + axiosError.message);
      }
    }
    throw new Error('An unexpected error occurred: ' + (error as Error).message);
  }
};

export const checkHealth = async (): Promise<HealthResponse> => {
  try {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      if (axiosError.response) {
        throw new Error(`Server error: ${axiosError.response.status}`);
      } else if (axiosError.request) {
        throw new Error('No response received from server. Please check if the backend is running.');
      } else {
        throw new Error('Request setup failed: ' + axiosError.message);
      }
    }
    throw new Error('An unexpected error occurred: ' + (error as Error).message);
  }
}; 