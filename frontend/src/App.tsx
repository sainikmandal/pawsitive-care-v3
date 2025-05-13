import { CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { PredictionForm } from './components/PredictionForm';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <PredictionForm />
    </ThemeProvider>
  );
}

export default App; 