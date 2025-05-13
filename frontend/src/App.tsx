import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Container, CssBaseline } from '@mui/material';
import { HomePage } from './components/HomePage';
import { PredictionForm } from './components/PredictionForm';
import { VetInformation } from './components/VetInformation';
import { StorePage } from './components/StorePage';

function App() {
  return (
    <Router>
      <CssBaseline />
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Pawsitive Care
          </Typography>
          <Button color="inherit" component={Link} to="/">
            Home
          </Button>
          <Button color="inherit" component={Link} to="/diagnosis">
            Diagnosis
          </Button>
          <Button color="inherit" component={Link} to="/vets">
            Find a Vet
          </Button>
          <Button color="inherit" component={Link} to="/shop">
            Pet Store
          </Button>
        </Toolbar>
      </AppBar>
      <Container>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/diagnosis" element={<PredictionForm />} />
          <Route path="/vets" element={<VetInformation />} />
          <Route path="/shop" element={<StorePage />} />
        </Routes>
      </Container>
    </Router>
  );
}

export default App; 