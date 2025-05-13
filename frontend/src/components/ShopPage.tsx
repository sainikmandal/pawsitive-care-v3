import { Box, Card, CardContent, CardMedia, Container, Grid, Typography, Button, Link } from '@mui/material';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';

interface Store {
  id: number;
  name: string;
  description: string;
  website: string;
  imageUrl: string;
  specialties: string[];
}

const TRUSTED_STORES: Store[] = [
  {
    id: 1,
    name: "PetSmart",
    description: "One of the largest pet supply retailers with a wide range of products for all types of pets.",
    website: "https://www.petsmart.com",
    imageUrl: "/stores/petsmart.jpg",
    specialties: ["Pet Food", "Toys", "Grooming Supplies", "Medications"]
  },
  {
    id: 2,
    name: "Chewy",
    description: "Online pet pharmacy and supply store with fast delivery and competitive prices.",
    website: "https://www.chewy.com",
    imageUrl: "/stores/chewy.jpg",
    specialties: ["Prescription Medications", "Pet Food", "Supplies", "Auto-Ship"]
  },
  {
    id: 3,
    name: "Petco",
    description: "Full-service pet supply store with veterinary services and grooming.",
    website: "https://www.petco.com",
    imageUrl: "/stores/petco.jpg",
    specialties: ["Pet Food", "Veterinary Services", "Grooming", "Training"]
  }
];

export const ShopPage = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Trusted Pet Supply Stores
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Browse through our curated list of reliable pet supply stores
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {TRUSTED_STORES.map((store) => (
          <Grid item key={store.id} xs={12} md={4}>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <CardMedia
                component="img"
                height="200"
                image={store.imageUrl}
                alt={store.name}
                sx={{ objectFit: 'cover' }}
              />
              <CardContent sx={{ flexGrow: 1 }}>
                <Typography gutterBottom variant="h5" component="h2">
                  {store.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {store.description}
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Specialties:
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {store.specialties.map((specialty) => (
                      <Typography
                        key={specialty}
                        variant="body2"
                        sx={{
                          bgcolor: 'primary.light',
                          color: 'primary.contrastText',
                          px: 1,
                          py: 0.5,
                          borderRadius: 1,
                        }}
                      >
                        {specialty}
                      </Typography>
                    ))}
                  </Box>
                </Box>
                <Button
                  variant="contained"
                  color="primary"
                  href={store.website}
                  target="_blank"
                  rel="noopener noreferrer"
                  startIcon={<ShoppingCartIcon />}
                  sx={{ mt: 2 }}
                >
                  Visit Store
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
}; 