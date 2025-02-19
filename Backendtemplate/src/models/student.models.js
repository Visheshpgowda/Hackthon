import mongoose, { Schema } from "mongoose";

const studentSchema = new Schema({
    name: {
        type: String,
        required: true,
        trim: true
    },
    roll_no: {
        type: Number,
        required: true,
        unique: true,
    },
    email: {
        type: String,
        required: true,
        unique: true,
    },
    phone_no: {
        type: Number,
        required: true,
        unique: true,
    },
    address: {
        type: String,
        required: true,
    },
    dob: {
        type: Date,
        required: true,
    },

}, {
    timestamps: true
})


export const Student = mongoose.model("Student", studentSchema)