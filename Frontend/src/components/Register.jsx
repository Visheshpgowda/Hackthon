import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UserPlus, Mail, Phone, Lock, Shield, AlertCircle, CheckCircle } from 'lucide-react';

function Register() {
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [phone, setPhone] = useState("");
    const [password, setPassword] = useState("");
    const [role, setRole] = useState("Teacher"); // Default role as "Teacher"
    const navigate = useNavigate();

    const handleSubmit = (e) => {
        e.preventDefault();
        console.log({ username, email, phone, password, role });
    };

    return (
        <div className="min-h-screen bg-orange-200 py-16 px-6 flex items-center justify-center">
            <div className="w-full max-w-lg">
                <div className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl p-10 space-y-8">
                    <div className="text-center">
                        <div className="mx-auto h-16 w-16 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full flex items-center justify-center">
                            <UserPlus className="h-8 w-8 text-white" />
                        </div>
                        <h2 className="mt-6 text-4xl font-bold text-gray-900">Create Account</h2>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Username */}
                        <div className="relative">
                            <label className="block text-base font-medium text-gray-700">Username</label>
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                                required
                            />
                        </div>

                        {/* Email */}
                        <div className="relative">
                            <label className="block text-base font-medium text-gray-700">Email</label>
                            <input
                                type="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                                required
                            />
                        </div>

                        {/* Phone */}
                        <div className="relative">
                            <label className="block text-base font-medium text-gray-700">Phone</label>
                            <input
                                type="tel"
                                value={phone}
                                onChange={(e) => setPhone(e.target.value)}
                                className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                                required
                            />
                        </div>

                        {/* Password */}
                        <div className="relative">
                            <label className="block text-base font-medium text-gray-700">Password</label>
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                className="w-full px-4 py-3 bg-white border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
                                required
                            />
                        </div>

                        {/* Role Selection */}
                        <div className="relative">
                            <label className="flex items-center space-x-3 cursor-pointer">
                                <input
                                    type="checkbox"
                                    checked={role === "Headmaster"}
                                    onChange={(e) => setRole(e.target.checked ? "Headmaster" : "Teacher")}
                                    className="w-5 h-5"
                                />
                                <span className="text-base font-medium text-gray-700">
                                    Register as Headmaster
                                </span>
                            </label>
                        </div>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            className="w-full py-3 rounded-lg text-lg font-medium text-white bg-purple-600 hover:bg-purple-700 transition"
                        >
                            Create Account
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
}

export default Register;
