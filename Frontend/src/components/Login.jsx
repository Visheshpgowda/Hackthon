import React, { useState } from 'react';
import axios from 'axios';
import { LogIn, Mail, Lock, UserPlus, Home, AlertCircle, CheckCircle, User } from 'lucide-react';

function Login() {
    const [email, setEmail] = useState("");
    const [name, setName] = useState(""); // Added name state
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");
    const [success, setSuccess] = useState("");
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setSuccess("");
        setLoading(true);

        try {
            const response = await axios.post('http://localhost:8080/api/v1/users/login', { email, name, password }, { withCredentials: true

             },{
                headers: {
                    'Content-Type': 'application/json',
                }
             }
             );
            
            setSuccess(response.data.message || "Logged in successfully!");
            console.log("User Data:", response.data.data.user);

            // You can navigate to the dashboard or home page after successful login
        } catch (err) {
            setError(err.response?.data?.message || "Failed to log in.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-orange-200 py-16 px-6 sm:px-8 lg:px-12 flex items-center justify-center">
            <div className="w-full max-w-lg mx-auto">
                <div className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl overflow-hidden p-10 space-y-8">
                    <div className="text-center">
                        <div className="mx-auto h-16 w-16 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full flex items-center justify-center">
                            <LogIn className="h-8 w-8 text-white" />
                        </div>
                        <h2 className="mt-6 text-4xl font-bold text-gray-900 tracking-tight">Welcome Back</h2>
                        <p className="mt-2 text-lg text-gray-600">Sign in to access your account</p>
                    </div>

                    {error && (
                        <div className="flex items-center p-4 bg-red-50 border-l-4 border-red-500 rounded-md">
                            <AlertCircle className="h-5 w-5 text-red-500 mr-3" />
                            <p className="text-red-700">{error}</p>
                        </div>
                    )}

                    {success && (
                        <div className="flex items-center p-4 bg-green-50 border-l-4 border-green-500 rounded-md">
                            <CheckCircle className="h-5 w-5 text-green-500 mr-3" />
                            <p className="text-green-700">{success}</p>
                        </div>
                    )}

                    <form className="space-y-8" onSubmit={handleSubmit}>
                        <div className="space-y-6">
                            {/* Name Field */}
                            <div className="relative">
                                <label className="block text-base font-medium text-gray-700 mb-2">Name</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <User className="h-6 w-6 text-gray-400" />
                                    </div>
                                    <input
                                        type="text"
                                        value={name}
                                        onChange={(e) => setName(e.target.value)}
                                        className="pl-12 w-full px-5 py-4 text-lg bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                                        required
                                        placeholder="Your Name"
                                    />
                                </div>
                            </div>

                            {/* Email Field */}
                            <div className="relative">
                                <label className="block text-base font-medium text-gray-700 mb-2">Email</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <Mail className="h-6 w-6 text-gray-400" />
                                    </div>
                                    <input
                                        type="email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        className="pl-12 w-full px-5 py-4 text-lg bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                                        required
                                        placeholder="your@email.com"
                                    />
                                </div>
                            </div>

                            {/* Password Field */}
                            <div className="relative">
                                <label className="block text-base font-medium text-gray-700 mb-2">Password</label>
                                <div className="relative">
                                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                        <Lock className="h-6 w-6 text-gray-400" />
                                    </div>
                                    <input
                                        type="password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        className="pl-12 w-full px-5 py-4 text-lg bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all"
                                        required
                                        placeholder="••••••••"
                                    />
                                </div>
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full flex justify-center py-4 px-5 border border-transparent rounded-lg shadow-sm text-lg font-medium text-white bg-purple-500 hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition-all"
                        >
                            {loading ? (
                                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            ) : (
                                "Sign in"
                            )}
                        </button>
                    </form>

                    <div className="space-y-6">
                        <div className="text-center">
                            <span className="text-base text-gray-600">Don't have an account?</span>
                        </div>
                        <div className="grid grid-cols-2 gap-6">
                            <button className="flex items-center justify-center py-3 px-5 rounded-lg text-base font-medium text-white bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 transition-all">
                                <UserPlus className="h-5 w-5 mr-2" />
                                Register
                            </button>
                            <button className="flex items-center justify-center py-3 px-5 rounded-lg text-base font-medium text-white bg-gradient-to-r from-red-500 to-rose-600 hover:from-red-600 hover:to-rose-700 transition-all">
                                <Home className="h-5 w-5 mr-2" />
                                Dashboard
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Login;
