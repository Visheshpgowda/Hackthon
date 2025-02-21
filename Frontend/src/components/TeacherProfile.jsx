import React, { useState, useEffect } from 'react';
import axios from 'axios';

function TeacherProfile() {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const response = await axios.get("http://localhost:8000/api/v1/users/profile", {
          withCredentials: true, // ensures cookies are sent
        });
        // Assuming response.data.data.user holds the teacher's info
        setProfile(response.data.data.user);
      } catch (err) {
        setError(err.response?.data?.message || "Failed to load profile.");
      } finally {
        setLoading(false);
      }
    };

    fetchProfile();
  }, []);

  if (loading) return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  if (error) return <div className="min-h-screen flex items-center justify-center text-red-600">Error: {error}</div>;

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-md mx-auto bg-white shadow-md rounded p-6">
        <h2 className="text-2xl font-bold mb-4">Teacher Profile</h2>
        <div className="space-y-4">
          <p>
            <strong>Name:</strong> {profile.name}
          </p>
          <p>
            <strong>Email:</strong> {profile.email}
          </p>
          <p>
            <strong>Phone:</strong> {profile.Phone_no}
          </p>
          <p>
            <strong>Role:</strong> {profile.role}
          </p>
        </div>
      </div>
    </div>
  );
}

export default TeacherProfile;
