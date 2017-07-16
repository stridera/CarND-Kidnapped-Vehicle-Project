/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

double addGaussianNoise(double mean, double std_dev) {
    default_random_engine gen;
    normal_distribution<double> dist(mean, std_dev);
    return dist(gen);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    num_particles = 100;

    std_x = 2;
    std_y = 2;
    std_theta = 0.05;

    for(int i = 0; i < num_particles; ++i) {
        Particle p = Particle();
        p.id = i;
        p.x = addGaussianNoise(x, std_x);
        p.y = addGaussianNoise(y, std_y);
        p.theta = addGaussianNoise(theta, std_theta);
        p.weight = 1;

        particles.push_back(p);
        weights.push_back(1.0);
    }
//
//    cout << "Particle List: " << endl;
//    for ( auto& p : particles ) {
//        cout << p.id << " (" << p.x << ", " << p.y << ")" << endl;
//    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    for ( auto& p : particles ) {

        // Update step from Project 2
        double yaw = p.theta;
        double yaw_p = yaw + yaw_rate * delta_t;
        if ( fabs(yaw_rate) > 0.0001 ) {
            p.x += velocity / yaw_rate * ( sin ( yaw_p ) - sin( p.theta ) );
            p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( yaw_p ) );
        }
        else {
            p.x += velocity * delta_t * cos( p.theta );
            p.y += velocity * delta_t * sin( p.theta );
        }
        p.theta = yaw_rate * delta_t;

        // Add Noise
        p.x = addGaussianNoise(p.x, std_pos[0]);
        p.y = addGaussianNoise(p.y, std_pos[0]);
        p.theta = addGaussianNoise(p.theta, std_pos[0]);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for ( auto& o : observations ) {
        double min = numeric_limits<double>::max();
        LandmarkObs closest;
        for ( auto& p : predicted) {
            double distance = dist(p.x, p.y, o.x, o.y);
            if ( distance < min ) {
                min = distance;
                closest = p;
                o.id = p.id;
            }
        }
        cout << "Obs: " << o.id << " (" << o.x << ", " << o.y << ") ==> (" << closest.x << ", " << closest.y << ")" << endl;
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
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // Pull this out to make it easier to read.  Will be used later.
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
        std::vector<LandmarkObs> map_obs;
        for ( auto& o : observations ) {
            double x = o.x * cos( p.theta ) - o.y * sin( p.theta ) + p.x;
            double y = o.x * sin( p.theta ) - o.y * cos( p.theta ) + p.y;

//            cout << "Adding Obs: " << o.id << " (" << x << ", " << y << ")" << endl;
            map_obs.push_back(LandmarkObs{o.id, x, y});
        }

        // Grab a list of landmarks that are within sensor range.  Make them landmark obs
        // for the data association function
        std::vector<LandmarkObs> predictions;
        for ( auto& point : map_landmarks.landmark_list ) {
            double dx = fabs(point.x_f - p.x);
            double dy = fabs(point.y_f - p.y);
            if (dx < sensor_range && dy < sensor_range) {
//                cout << "Adding Prediction: " << point.id_i << " (" << point.x_f << ", " << point.y_f << ")" << endl;
                predictions.push_back(LandmarkObs{point.id_i, point.x_f, point.y_f});
            }
        }

        // Next each measurement will need to be associated with a landmark identifier,
        // for doing this part we will simply take the closest landmark to each transformed observation.
        // Finally we will then have everything we need to calculate the particles weight value.
        dataAssociation(predictions, map_obs);

        // Now we that we have done the measurement transformations and associations, we have all the pieces
        // we need to calculate the particle's final weight. The particles final weight will be calculated as
        // the product of each measurement's Multivariate-Gaussian probability.
        // Vars needed for the SetAccociations function:
        std:vector<int> associations;
        std::vector<double> sense_x;
        std::vector<double> sense_y;

        double weight = 1;
        for ( auto& o : map_obs) {
            // Formula help from https://discussions.udacity.com/t/transformations-and-associations-calculating-the-particles-final-weight/308602/3
            auto pred = find_if(begin(predictions), end(predictions), [o](LandmarkObs const& l) {
                return (l.id == o.id);
            });
            cout << "Obs->Pred: " << o.id << " (" << o.x << ", " << o.y << ") - " <<
                     pred->id << ": (" << pred->x << ", " << pred->y << ") - " <<
                     endl;

            double pow_x = pow(pred->x - o.x, 2.0);
            double pow_y = pow(pred->y - o.y, 2.0);
            double ev = ( -pow_x / ( 2.0 * std_x_2 ) ) - ( pow_y / ( 2.0 * std_y_2 ) );

            double norm = exp( ev ) / measure;
            cout <<
                 " ev: " << ev <<
//                 " std_x_2: " << std_x_2 <<
//                 " std_y_2: " << std_y_2 <<
                 " pow_x: " << pow_x <<
                 " pow_y: " << pow_y <<
//                 " std_x: " << std_x <<
//                 " std_y: " << std_y <<
                 " Norm: " << norm <<
                 endl;

            if (norm > 0) {
                weight *= norm;
            }
            associations.push_back(o.id);
            sense_x.push_back(o.x);
            sense_y.push_back(o.y);
        }

        cout << "Particle " << p.id << " old weight: " << p.weight << " new weight: " << weight << endl;
        p.weight = weight;
        SetAssociations(p, associations, sense_x, sense_y);
    }
//    exit(1);

}

void ParticleFilter::resample() {
    weights.clear();
    for (auto& p : particles) {
//        cout << "Adding Weight from particle " << p.id << " - " << p.weight << endl;
        weights.push_back(p.weight);
    }

    discrete_distribution<double> d(weights.begin(), weights.end());

    std::vector<Particle> resampled_particles;
    default_random_engine gen;

    for (auto& p : particles) {
        double id = d(gen);
        Particle selected = particles[id];
//        cout << "Selected: " << selected.id << " (" << selected.x << ", " << selected.y << ") - " << id << endl;

        resampled_particles.push_back(selected);
    }

    particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
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
