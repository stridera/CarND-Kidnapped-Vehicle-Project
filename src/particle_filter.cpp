/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <sstream>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 100;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for(int i = 0; i < num_particles; ++i) {
        Particle p = Particle();
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;

        particles.push_back(p);
        weights.push_back(1.0);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    // Add Noise
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for ( auto& p : particles ) {
        double x, y;

        // Update step from Project 2
        double yaw = p.theta;
        double yaw_p = yaw + (yaw_rate * delta_t);
        if ( fabs(yaw_rate) > 0.0001 ) {
            p.x += (velocity / yaw_rate) * ( sin ( yaw_p ) - sin( yaw ) );
            p.y += (velocity / yaw_rate) * ( cos( yaw ) - cos( yaw_p ) );
        }
        else {
            p.x += velocity * delta_t * cos( yaw );
            p.y += velocity * delta_t * sin( yaw );
        }
        p.theta += yaw_rate * delta_t;

        // Add Noise
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) {
    for ( auto& o : observations ) {
        double min = numeric_limits<double>::max();
        for ( auto& p : predicted) {
            double distance = dist(p.x, p.y, o.x, o.y);
            if ( distance < min ) {
                min = distance;
                o.id = p.id;
            }
        }
    }
}

/**
 *
 * @param sensor_range - Sensor range [m]
 * @param std_landmark - Landmark measurement uncertainty [x [m], y [m]]
 * @param observations - noisy observation data from the simulator
 * @param map_landmarks - Dora says this is THE MAP!
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		vector<LandmarkObs> observations, Map map_landmarks) {

    // Pulling out common vars to make it easier to read and speed up processing.  Will be used later.
    double std_x = std_landmark[0];
    double std_y = std_landmark[1];
    double std_x_2 = pow(std_x, 2.0);
    double std_y_2 = pow(std_y, 2.0);
    double measure = 2 * M_PI * std_x * std_y;

    // Lesson 14.13
    // Take a single particle with its position and heading along with the cars observation measurements.
    for ( auto& p : particles ) {
        // We will first need to transform the car's measurements from its local car coordinate system to the
        // map's coordinate system.  (x cos θ - y sin θ + xt), (x sin θ + y cos θ + yt)
        vector<LandmarkObs> translated_observations;
        for ( auto& o : observations ) {
            double x = p.x + o.x * cos( p.theta ) - o.y * sin( p.theta );
            double y = p.y + o.x * sin( p.theta ) + o.y * cos( p.theta );

            translated_observations.push_back(LandmarkObs{o.id, x, y});
        }

        // Grab a list of landmarks that are within sensor range.  Make them landmark obs
        // for the data association function
        vector<LandmarkObs> nearby_landmarks;
        for ( auto& lm : map_landmarks.landmark_list ) {
            double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
            if (distance < sensor_range) {
                nearby_landmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
            }
        }

        if (nearby_landmarks.size() == 0) {
            p.weight = 0.0;
            continue;
        }

        // Next each measurement will need to be associated with a landmark identifier,
        // for doing this part we will simply take the closest landmark to each transformed observation.
        // Finally we will then have everything we need to calculate the particles weight value.
        dataAssociation(nearby_landmarks, translated_observations);

        // Now we that we have done the measurement transformations and associations, we have all the pieces
        // we need to calculate the particle's final weight. The particles final weight will be calculated as
        // the product of each measurement's Multivariate-Gaussian probability.

        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;

        double weight = 1;
        for ( auto& o : translated_observations) {
            auto pred = find_if(begin(nearby_landmarks), end(nearby_landmarks), [o](LandmarkObs const& l) {
                return (l.id == o.id);
            });

            double pow_x = pow(pred->x - o.x, 2.0);
            double pow_y = pow(pred->y - o.y, 2.0);
            double prob = exp( ( -pow_x / ( 2.0 * std_x_2 ) ) - ( pow_y / ( 2.0 * std_y_2 ) ) ) / measure;

            weight *= prob;
            associations.push_back(o.id);
            sense_x.push_back(o.x);
            sense_y.push_back(o.y);
        }

        p.weight = weight;
        SetAssociations(p, associations, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
    weights.clear();
    for (auto& p : particles) {
        weights.push_back(p.weight);
    }

    discrete_distribution<double> d(weights.begin(), weights.end());

    vector<Particle> resampled_particles;
    default_random_engine gen;

    for (int i = 0; i < particles.size(); i++) {
        resampled_particles.push_back(particles[d(gen)]);
    }
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, vector<int> associations, vector<double> sense_x, vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
