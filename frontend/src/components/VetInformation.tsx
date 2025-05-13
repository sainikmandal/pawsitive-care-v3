import { Box, Card, CardContent, CardMedia, Grid, Typography, Container, Button, Tabs, Tab } from '@mui/material';
import { useState } from 'react';
import LocalHospitalIcon from '@mui/icons-material/LocalHospital';
import VideoCallIcon from '@mui/icons-material/VideoCall';

interface VetInfo {
  id: number;
  name: string;
  specialization: string;
  experience: string;
  location: string;
  contact: string;
  imageUrl: string;
  description: string;
}

interface OnlineVetService {
  id: number;
  name: string;
  description: string;
  website: string;
  imageUrl: string;
  features: string[];
}

const TRUSTED_VETS: VetInfo[] = [
  {
    id: 1,
    name: "DR. SHUBHAMITRA CHAUDHURI",
    specialization: "Large Animal Specialist",
    experience: "15 years",
    location: "Kolkata, West Bengal",
    contact: "+91 9874579755",
    imageUrl: "/vets/vet1.jpg",
    description: "Specializes in cattle and livestock diseases with extensive experience in emergency care."
  },
  {
    id: 2,
    name: "Dr. Rashmi Singh",
    specialization: "Livestock Health Expert",
    experience: "12 years",
    location: "Kolkata, West Bengal",
    contact: "+91 98300 00000",
    imageUrl: "/vets/vet2.jpg",
    description: "Expert in preventive care and disease management for farm animals."
  },
  {
    id: 3,
    name: "DR. SAMIT KUMAR NANDI",
    specialization: "Emergency Care Specialist",
    experience: "10 years",
    location: "Kolkata, West Bengal",
    contact: "+91 9433111065",
    imageUrl: "/vets/vet3.jpg",
    description: "24/7 emergency services with mobile veterinary unit for farm visits."
  }
];

const ONLINE_VET_SERVICES: OnlineVetService[] = [
  {
    id: 1,
    name: "PetCoach",
    description: "24/7 online veterinary consultation service with experienced vets.",
    website: "https://www.petcoach.co",
    imageUrl: "/online-vets/petcoach.jpg",
    features: ["24/7 Availability", "Video Consultations", "Prescription Services", "Follow-up Care"]
  },
  {
    id: 2,
    name: "FirstVet",
    description: "Immediate access to licensed veterinarians through video calls.",
    website: "https://firstvet.com",
    imageUrl: "/online-vets/firstvet.jpg",
    features: ["Quick Response", "Multi-language Support", "Emergency Care", "Health Records"]
  },
  {
    id: 3,
    name: "Vetster",
    description: "Connect with licensed veterinarians for online consultations.",
    website: "https://vetster.com",
    imageUrl: "/online-vets/vetster.jpg",
    features: ["On-demand Consultations", "Specialist Access", "Prescription Services", "Health Plans"]
  }
];

export const VetInformation = () => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h4" component="h1" gutterBottom align="center">
        Veterinary Services
      </Typography>
      <Typography variant="subtitle1" gutterBottom align="center" color="text.secondary" sx={{ mb: 4 }}>
        Choose between local veterinarians or online consultation services
      </Typography>

      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 4 }}>
        <Tabs value={activeTab} onChange={handleTabChange} centered>
          <Tab 
            icon={<LocalHospitalIcon />} 
            label="Local Veterinarians" 
            iconPosition="start"
          />
          <Tab 
            icon={<VideoCallIcon />} 
            label="Online Services" 
            iconPosition="start"
          />
        </Tabs>
      </Box>

      {activeTab === 0 ? (
        <Grid container spacing={4}>
          {TRUSTED_VETS.map((vet) => (
            <Grid item key={vet.id} xs={12} sm={6} md={4}>
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardMedia
                  component="img"
                  height="200"
                  image={vet.imageUrl}
                  alt={vet.name}
                  sx={{ objectFit: 'cover' }}
                />
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography gutterBottom variant="h5" component="h2">
                    {vet.name}
                  </Typography>
                  <Typography variant="subtitle1" color="primary" gutterBottom>
                    {vet.specialization}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {vet.description}
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      <strong>Experience:</strong> {vet.experience}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Location:</strong> {vet.location}
                    </Typography>
                    <Typography variant="body2">
                      <strong>Contact:</strong> {vet.contact}
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
        <Grid container spacing={4}>
          {ONLINE_VET_SERVICES.map((service) => (
            <Grid item key={service.id} xs={12} sm={6} md={4}>
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardMedia
                  component="img"
                  height="200"
                  image={service.imageUrl}
                  alt={service.name}
                  sx={{ objectFit: 'cover' }}
                />
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography gutterBottom variant="h5" component="h2">
                    {service.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {service.description}
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Features:
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {service.features.map((feature) => (
                        <Typography
                          key={feature}
                          variant="body2"
                          sx={{
                            bgcolor: 'primary.light',
                            color: 'primary.contrastText',
                            px: 1,
                            py: 0.5,
                            borderRadius: 1,
                          }}
                        >
                          {feature}
                        </Typography>
                      ))}
                    </Box>
                  </Box>
                  <Button
                    variant="contained"
                    color="primary"
                    href={service.website}
                    target="_blank"
                    rel="noopener noreferrer"
                    startIcon={<VideoCallIcon />}
                    sx={{ mt: 2 }}
                  >
                    Visit Service
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}
    </Container>
  );
}; 