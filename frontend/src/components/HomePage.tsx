import { Box, Card, CardContent, CardMedia, Container, Grid, Typography, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import PetsIcon from '@mui/icons-material/Pets';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';

const FEATURES = [
  {
    id: 'diagnosis',
    title: 'ML-Powered Diagnosis',
    description: 'Get instant disease predictions for your livestock using our advanced machine learning model.',
    icon: <PetsIcon sx={{ fontSize: 60 }} />,
    path: '/diagnosis',
    color: '#1976d2'
  },
  {
    id: 'vets',
    title: 'Find a Veterinarian',
    description: 'Connect with trusted veterinarians in your area or consult online veterinary services.',
    icon: <LocalHospitalIcon sx={{ fontSize: 60 }} />,
    path: '/vets',
    color: '#2e7d32'
  },
  {
    id: 'shop',
    title: 'Shop for Your Pet',
    description: 'Browse through trusted online stores for pet supplies, medicines, and accessories.',
    icon: <ShoppingCartIcon sx={{ fontSize: 60 }} />,
    path: '/shop',
    color: '#ed6c02'
  }
];

export const HomePage = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="lg">
      {/* Header Section */}
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Typography variant="h2" component="h1" gutterBottom>
          Welcome to Pawsitive Care
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          Your one-stop solution for livestock health and care
        </Typography>
      </Box>

      {/* Features Grid */}
      <Grid container spacing={4} sx={{ mb: 8 }}>
        {FEATURES.map((feature) => (
          <Grid item xs={12} md={4} key={feature.id}>
            <Card 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: 6
                }
              }}
            >
              <CardContent sx={{ 
                flexGrow: 1, 
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
                p: 4
              }}>
                <Box sx={{ color: feature.color, mb: 2 }}>
                  {feature.icon}
                </Box>
                <Typography variant="h4" component="h2" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph sx={{ mb: 3 }}>
                  {feature.description}
                </Typography>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => navigate(feature.path)}
                  sx={{ 
                    mt: 'auto',
                    bgcolor: feature.color,
                    '&:hover': {
                      bgcolor: feature.color,
                      opacity: 0.9
                    }
                  }}
                >
                  Get Started
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}; 