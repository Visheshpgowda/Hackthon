import mongoose, { Schema } from 'mongoose';

const subjectSchema = new Schema({
    Subject_name: {
        type: String,
        required: true
    },
    teacher: [{
        type: Schema.Types.ObjectId,
        ref: 'User',
        required: true
    }],
});
export const Subject = mongoose.model('Subject', subjectSchema);

