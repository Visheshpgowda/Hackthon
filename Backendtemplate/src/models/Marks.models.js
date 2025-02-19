import mongoose, { Schema } from "mongoose";

const marksSchema = new Schema({
    Student_name: {
        type: Schema.Types.ObjectId,
        ref: 'Student',
        required: true
    },
    roll_no: {
        type: Number,
        required: true
    },
    subject: {
        type: String,
        required: true
    },
    Exam_type: {
        type: String,
        enum: ['Midterm', 'Final'],
        required: true
    },
    marks: {
        type: Number,
        required: true
    },
    teacher: {
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    }
}, {
    timestamps: true
});


export const Marks = mongoose.model('Marks', marksSchema);